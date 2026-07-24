//===-- Implementation of libc death test executors -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/stdint_proxy.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include <stddef.h>

#ifdef LIBC_TARGET_ARCH_IS_AARCH64
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

extern "C" {

// Hermetic tests rely on the following memory functions. This is because the
// compiler code generation can emit calls to them. We want to map the external
// entrypoint to the internal implementation of the function used for testing.
// This is done manually as not all targets support aliases.

int bcmp(const void *lhs, const void *rhs, size_t count) {
  return LIBC_NAMESPACE::bcmp(lhs, rhs, count);
}
void bzero(void *ptr, size_t count) { LIBC_NAMESPACE::bzero(ptr, count); }
int memcmp(const void *lhs, const void *rhs, size_t count) {
  return LIBC_NAMESPACE::memcmp(lhs, rhs, count);
}
void *memcpy(void *__restrict dst, const void *__restrict src, size_t count) {
  return LIBC_NAMESPACE::memcpy(dst, src, count);
}
void *memmove(void *dst, const void *src, size_t count) {
  return LIBC_NAMESPACE::memmove(dst, src, count);
}
void *memset(void *ptr, int value, size_t count) {
  return LIBC_NAMESPACE::memset(ptr, value, count);
}

// This is needed if the test was compiled with '-fno-use-cxa-atexit'.
int atexit(void (*func)(void)) { return LIBC_NAMESPACE::atexit(func); }

void *malloc(size_t s) noexcept;
void free(void *ptr) noexcept;
void *realloc(void *mem, size_t s) noexcept;

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

#ifdef LIBC_TARGET_ARCH_IS_AARCH64
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

void operator delete(void *ptr, size_t) { free(ptr); }

// Defining members in the std namespace is not preferred. But, we do it here
// so that we can use it to define the operator new which takes std::align_val_t
// argument.
namespace std {
enum class align_val_t : size_t {};
} // namespace std

void operator delete(void *ptr, std::align_val_t) noexcept { free(ptr); }

void operator delete(void *ptr, unsigned int, std::align_val_t) noexcept {
  free(ptr);
}
