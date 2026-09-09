//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Hermetic test runtime utilities, allocator stubs, and compiler runtime
/// hooks.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include <stddef.h>

#if defined(LIBC_TARGET_OS_IS_LINUX)
#include "src/__support/OSUtil/linux/syscall.h"
#include <sys/syscall.h>
#endif

#if defined(LIBC_TARGET_ARCH_IS_AARCH64) &&                                    \
    !defined(LIBC_TARGET_OS_IS_BAREMETAL)
#include "src/sys/auxv/getauxval.h"
#endif

namespace LIBC_NAMESPACE_DECL {

int bcmp(const void *lhs, const void *rhs, size_t count);
void bzero(void *ptr, size_t count);
int memcmp(const void *lhs, const void *rhs, size_t count);
void *memcpy(void *__restrict, const void *__restrict, size_t);
void *memmove(void *dst, const void *src, size_t count);
void *memset(void *ptr, int value, size_t count);
int atexit(void (*func)(void));

} // namespace LIBC_NAMESPACE_DECL

constexpr uint64_t ALIGNMENT = alignof(uintptr_t);

namespace {

// Integration tests cannot use the SCUDO standalone allocator as SCUDO pulls
// various other parts of the libc. Since SCUDO development does not use
// LLVM libc build rules, it is very hard to keep track or pull all that SCUDO
// requires. Hence, as a work around for this problem, we use a simple allocator
// which just hands out continuous blocks from a statically allocated chunk of
// memory.
static constexpr uint64_t MEMORY_SIZE = 1 << 20; // 1 MiB
alignas(ALIGNMENT) static uint8_t memory[MEMORY_SIZE];
static uint8_t *ptr = memory;

} // anonymous namespace

extern "C" {

// Hermetic tests rely on the following memory functions. This is because the
// compiler code generation can emit calls to them. We want to map the external
// entrypoint to the internal implementation of the function used for testing.
// This is done manually as not all targets support aliases.

[[gnu::weak]] int bcmp(const void *lhs, const void *rhs, size_t count) {
  return LIBC_NAMESPACE::bcmp(lhs, rhs, count);
}
[[gnu::weak]] void bzero(void *ptr, size_t count) {
  LIBC_NAMESPACE::bzero(ptr, count);
}
[[gnu::weak]] int memcmp(const void *lhs, const void *rhs, size_t count) {
  return LIBC_NAMESPACE::memcmp(lhs, rhs, count);
}
[[gnu::weak]] void *memcpy(void *__restrict dst, const void *__restrict src,
                           size_t count) {
  return LIBC_NAMESPACE::memcpy(dst, src, count);
}
[[gnu::weak]] void *memmove(void *dst, const void *src, size_t count) {
  return LIBC_NAMESPACE::memmove(dst, src, count);
}
[[gnu::weak]] void *memset(void *ptr, int value, size_t count) {
  return LIBC_NAMESPACE::memset(ptr, value, count);
}

// This is needed if the test was compiled with '-fno-use-cxa-atexit'.
[[gnu::weak]] int atexit(void (*func)(void)) {
  return LIBC_NAMESPACE::atexit(func);
}

[[gnu::weak]] void *aligned_alloc(size_t align, size_t s) {
  if (align & (align - 1)) // Must be power of 2
    return nullptr;
  uintptr_t ptr_val = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t aligned_ptr_val = ((ptr_val + align - 1) / align) * align;
  ptr = reinterpret_cast<uint8_t *>(aligned_ptr_val);
  void *mem = ptr;
  ptr += s;
  return static_cast<uint64_t>(ptr - memory) >= MEMORY_SIZE ? nullptr : mem;
}

[[gnu::weak]] void *malloc(size_t s) { return aligned_alloc(ALIGNMENT, s); }

/// Allocates zero-initialized memory for hermetic test execution.
/// Satisfies runtime memory dependencies referenced by libclang_rt.profile.a.
///
/// \param num Number of elements.
/// \param size Size of each element in bytes.
/// \return Pointer to zero-initialized allocated memory, or nullptr on failure.
[[gnu::weak]] void *calloc(size_t num, size_t size) {
  if (num == 0 || size == 0)
    return nullptr;
  size_t total;
  if (__builtin_mul_overflow(num, size, &total)) {
    libc_errno = ENOMEM;
    return nullptr;
  }
  void *mem = malloc(total);
  if (mem == nullptr) {
    libc_errno = ENOMEM;
    return nullptr;
  }
  LIBC_NAMESPACE::memset(mem, 0, total);
  return mem;
}

[[gnu::weak]] void free(void *) {}

#if defined(__linux__)
/// Bridges compiler-rt profiling errno accesses to LLVM-libc thread-local
/// errno.
extern "C" [[gnu::const]] int *__errno_location() noexcept {
  return LIBC_NAMESPACE::__llvm_libc_errno();
}
#endif

[[gnu::weak]] void *realloc(void *mem, size_t s) {
  if (mem == nullptr)
    return malloc(s);
  uint8_t *newmem = reinterpret_cast<uint8_t *>(malloc(s));
  if (newmem == nullptr)
    return nullptr;
  uint8_t *oldmem = reinterpret_cast<uint8_t *>(mem);
  // We use a simple for loop to copy the data over.
  // If |s| is less the previous alloc size, the copy works as expected.
  // If |s| is greater than the previous alloc size, then garbage is copied
  // over to the additional part in the new memory block.
  for (size_t i = 0; i < s; ++i)
    newmem[i] = oldmem[i];
  return newmem;
}

void *calloc(size_t num, size_t size) {
  size_t total;
  if (__builtin_mul_overflow(num, size, &total))
    return nullptr;
  void *mem = malloc(total);
  if (mem != nullptr)
    LIBC_NAMESPACE::memset(mem, 0, total);
  return mem;
}

int *__llvm_libc_errno() noexcept;
int *__errno_location() { return __llvm_libc_errno(); }

#if defined(LIBC_TARGET_OS_IS_LINUX)
__attribute__((constructor)) static void __clean_hermetic_environment() {
  for (int fd = 3; fd < 256; ++fd)
    LIBC_NAMESPACE::syscall_impl<long>(SYS_close, fd);
  *__llvm_libc_errno() = 0;
}
#endif

// The unit test framework uses pure virtual functions. Since hermetic tests
// cannot depend C++ runtime libraries, implement dummy functions to support
// the virtual function runtime.
void __cxa_pure_virtual() {
  // A pure virtual being called is an error so we just trap.
  __builtin_trap();
}

// Hermetic tests are linked with -nostdlib. BFD linker expects
// __dso_handle when -nostdlib is used.
void *__dso_handle = nullptr;

#if defined(LIBC_TARGET_ARCH_IS_AARCH64) &&                                    \
    !defined(LIBC_TARGET_OS_IS_BAREMETAL)
// Due to historical reasons, libgcc on aarch64 may expect __getauxval to be
// defined. See also https://gcc.gnu.org/pipermail/gcc-cvs/2020-June/300635.html
unsigned long __getauxval(unsigned long id) {
  return LIBC_NAMESPACE::getauxval(id);
}
#endif

} // extern "C"

void *operator new([[maybe_unused]] size_t size, void *ptr) { return ptr; }

void *operator new(size_t size) { return malloc(size); }

void *operator new[](size_t size) { return malloc(size); }

void operator delete(void *ptr) { free(ptr); }

void operator delete[](void *ptr) { free(ptr); }

void operator delete(void *ptr, size_t) { free(ptr); }

// Defining members in the std namespace is not preferred. But, we do it here
// so that we can use it to define the operator new which takes std::align_val_t
// argument.
namespace std {
enum class align_val_t : size_t {};
} // namespace std

void operator delete(void *ptr, std::align_val_t) noexcept { free(ptr); }

void operator delete(void *ptr, size_t, std::align_val_t) noexcept {
  free(ptr);
}
