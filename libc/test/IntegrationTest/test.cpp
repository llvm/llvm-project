//===-- Simple malloc and free for use with integration tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stddef.h>
#include <stdint.h>

// Integration tests rely on the following memory functions. This is because the
// compiler code generation can emit calls to them. We want to map the external
// entrypoint to the internal implementation of the function used for testing.
// This is done manually as not all targets support aliases.

namespace __llvm_libc {

int bcmp(const void *lhs, const void *rhs, size_t count);
void bzero(void *ptr, size_t count);
int memcmp(const void *lhs, const void *rhs, size_t count);
void *memcpy(void *__restrict, const void *__restrict, size_t);
void *memmove(void *dst, const void *src, size_t count);
void *memset(void *ptr, int value, size_t count);
int atexit(void (*func)(void));

} // namespace __llvm_libc

extern "C" {

int bcmp(const void *lhs, const void *rhs, size_t count) {
  return __llvm_libc::bcmp(lhs, rhs, count);
}
void bzero(void *ptr, size_t count) { __llvm_libc::bzero(ptr, count); }
int memcmp(const void *lhs, const void *rhs, size_t count) {
  return __llvm_libc::memcmp(lhs, rhs, count);
}
void *memcpy(void *__restrict dst, const void *__restrict src, size_t count) {
  return __llvm_libc::memcpy(dst, src, count);
}
void *memmove(void *dst, const void *src, size_t count) {
  return __llvm_libc::memmove(dst, src, count);
}
void *memset(void *ptr, int value, size_t count) {
  return __llvm_libc::memset(ptr, value, count);
}

// This is needed if the test was compiled with '-fno-use-cxa-atexit'.
int atexit(void (*func)(void)) { return __llvm_libc::atexit(func); }

} // extern "C"

// Integration tests cannot use the SCUDO standalone allocator as SCUDO pulls
// various other parts of the libc. Since SCUDO development does not use
// LLVM libc build rules, it is very hard to keep track or pull all that SCUDO
// requires. Hence, as a work around for this problem, we use a simple allocator
// which just hands out continuous blocks from a statically allocated chunk of
// memory.

static constexpr uint64_t MEMORY_SIZE = 16384;
static uint8_t memory[MEMORY_SIZE];
static uint8_t *ptr = memory;

extern "C" {

void *malloc(size_t s) {
  void *mem = ptr;
  ptr += s;
  return static_cast<uint64_t>(ptr - memory) >= MEMORY_SIZE ? nullptr : mem;
}

void free(void *) {}

void *realloc(void *ptr, size_t s) {
  free(ptr);
  return malloc(s);
}

// Integration tests are linked with -nostdlib. BFD linker expects
// __dso_handle when -nostdlib is used.
void *__dso_handle = nullptr;
} // extern "C"
