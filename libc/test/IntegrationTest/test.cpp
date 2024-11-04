//===-- Simple malloc and free for use with integration tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include <stddef.h>
#include <stdint.h>

#ifdef LIBC_TARGET_ARCH_IS_AARCH64
#include "src/sys/auxv/getauxval.h"
#endif

// Integration tests rely on the following memory functions. This is because the
// compiler code generation can emit calls to them. We want to map the external
// entrypoint to the internal implementation of the function used for testing.
// This is done manually as not all targets support aliases.

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

#ifdef LIBC_TARGET_ARCH_IS_AARCH64
// Due to historical reasons, libgcc on aarch64 may expect __getauxval to be
// defined. See also https://gcc.gnu.org/pipermail/gcc-cvs/2020-June/300635.html
unsigned long __getauxval(unsigned long id) {
  return LIBC_NAMESPACE::getauxval(id);
}
#endif
} // extern "C"
