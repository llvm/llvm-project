//===-- Implementation of libc death test executors -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stddef.h>
#include <stdint.h>

namespace LIBC_NAMESPACE {

int bcmp(const void *lhs, const void *rhs, size_t count);
void bzero(void *ptr, size_t count);
int memcmp(const void *lhs, const void *rhs, size_t count);
void *memcpy(void *__restrict, const void *__restrict, size_t);
void *memmove(void *dst, const void *src, size_t count);
void *memset(void *ptr, int value, size_t count);
int atexit(void (*func)(void));

} // namespace LIBC_NAMESPACE

namespace {

// Integration tests cannot use the SCUDO standalone allocator as SCUDO pulls
// various other parts of the libc. Since SCUDO development does not use
// LLVM libc build rules, it is very hard to keep track or pull all that SCUDO
// requires. Hence, as a work around for this problem, we use a simple allocator
// which just hands out continuous blocks from a statically allocated chunk of
// memory.
static constexpr uint64_t MEMORY_SIZE = 65336;
static uint8_t memory[MEMORY_SIZE];
static uint8_t *ptr = memory;

} // anonymous namespace

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

constexpr uint64_t ALIGNMENT = alignof(uintptr_t);

void *malloc(size_t s) {
  // Keep the bump pointer aligned on an eight byte boundary.
  s = ((s + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
  void *mem = ptr;
  ptr += s;
  return static_cast<uint64_t>(ptr - memory) >= MEMORY_SIZE ? nullptr : mem;
}

void free(void *) {}

void *realloc(void *mem, size_t s) {
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

} // extern "C"

void *operator new(size_t size) { return malloc(size); }

void *operator new[](size_t size) { return malloc(size); }

void operator delete(void *) {
  // The libc runtime should not use the global delete operator. Hence,
  // we just trap here to catch any such accidental usages.
  __builtin_trap();
}
