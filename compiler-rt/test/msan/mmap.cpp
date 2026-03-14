// Test that mmap (without MAP_FIXED) always returns valid application addresses.
// RUN: %clangxx_msan -O0 %s -o %t && %run %t
// RUN: %clangxx_msan -O0 -fsanitize-memory-track-origins %s -o %t && %run %t

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>
#include "test.h"

bool AddrIsApp(void *p) {
  uintptr_t addr = (uintptr_t)p;
#if defined(__FreeBSD__) && defined(__x86_64__)
  return addr < 0x010000000000ULL || addr >= 0x600000000000ULL;
#elif defined(__x86_64__)
  return (addr >= 0x000000000000ULL && addr < 0x010000000000ULL) ||
         (addr >= 0x510000000000ULL && addr < 0x600000000000ULL) ||
         (addr >= 0x700000000000ULL && addr < 0x800000000000ULL);
#elif defined(__loongarch_lp64)
  return (addr >= 0x000000000000ULL && addr < 0x010000000000ULL) ||
         (addr >= 0x510000000000ULL && addr < 0x600000000000ULL) ||
         (addr >= 0x700000000000ULL && addr < 0x800000000000ULL);
#elif defined(__mips64)
  return (addr >= 0x0000000000ULL && addr <= 0x0200000000ULL) ||
         (addr >= 0xa200000000ULL && addr <= 0xc000000000ULL) ||
         addr >= 0xe200000000ULL;
#elif defined(__powerpc64__)
  return addr < 0x000100000000ULL || addr >= 0x300000000000ULL;
#elif defined(__s390x__)
  return addr < 0x040000000000ULL ||
         (addr >= 0x440000000000ULL && addr < 0x500000000000);
#elif defined(__aarch64__)

  struct AddrMapping {
    uintptr_t start;
    uintptr_t end;
  } mappings[] = {
      {0x0000000000000ULL, 0x0100000000000ULL},
      {0x0A00000000000ULL, 0x0B00000000000ULL},
      {0x0E00000000000ULL, 0x0F00000000000ULL},
      {0x0F00000000000ULL, 0x1000000000000ULL},
  };
  const size_t mappingsSize = sizeof (mappings) / sizeof (mappings[0]);

  for (int i=0; i<mappingsSize; ++i)
    if (addr >= mappings[i].start && addr < mappings[i].end)
      return true;
  return false;
#endif
}

int main() {
  // Large enough to quickly exhaust the entire address space.
#if defined(__mips64) || defined(__aarch64__)
  const size_t kMapSize = 0x100000000ULL;
#else
  const size_t kMapSize = 0x1000000000ULL;
#endif
  int success_count = 0;
  int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#if defined(MAP_NORESERVE)
  flags |= MAP_NORESERVE;
#endif
  while (true) {
    void *p = mmap(0, kMapSize, PROT_WRITE,
                   flags, -1, 0);
    printf("%p\n", p);
    if (p == MAP_FAILED) {
      assert(errno == ENOMEM);
      break;
    }
    assert(AddrIsApp(p));
    success_count++;
  }
  printf("successful mappings: %d\n", success_count);
  assert(success_count > 5);
}
