// RUN: %clangxx_msan -O0 %s -o %t && %run %t

// Check that strlen() and similar intercepted functions can be called on shadow
// memory.
// The mem_to_shadow's part might need rework
// XFAIL: freebsd

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "test.h"

const char *mem_to_shadow(const char *p) {
#if defined(__x86_64__)
  return (char *)((uintptr_t)p ^ 0x500000000000ULL);
#elif defined (__mips64)
  return (char *)((uintptr_t)p ^ 0x8000000000ULL);
#elif defined(__powerpc64__)
#define LINEARIZE_MEM(mem) \
  (((uintptr_t)(mem) & ~0x200000000000ULL) ^ 0x100000000000ULL)
  return (char *)(LINEARIZE_MEM(p) + 0x080000000000ULL);
#elif defined(__aarch64__)
  return (char *)((uintptr_t)p ^ 0x6000000000ULL);
#endif
}

int main(void) {
  const char *s = "abcdef";
  assert(strlen(s) == 6);
  assert(strlen(mem_to_shadow(s)) == 0);

  char *t = new char[42];
  t[41] = 0;
  assert(strlen(mem_to_shadow(t)) == 41);
  return 0;
}
