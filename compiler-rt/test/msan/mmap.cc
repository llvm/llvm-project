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
#elif defined(__mips64)
  return (addr >= 0x0000000000ULL && addr <= 0x0200000000ULL) ||
         (addr >= 0xa200000000ULL && addr <= 0xc000000000ULL) ||
         addr >= 0xe200000000ULL;
#elif defined(__powerpc64__)
  return addr < 0x000100000000ULL || addr >= 0x300000000000ULL;
#elif defined(__aarch64__)

  struct AddrMapping {
    uintptr_t start;
    uintptr_t end;
  } mappings[] = {
    {0x05000000000ULL, 0x06000000000ULL},
    {0x07000000000ULL, 0x08000000000ULL},
    {0x0F000000000ULL, 0x10000000000ULL},
    {0x11000000000ULL, 0x12000000000ULL},
    {0x20000000000ULL, 0x21000000000ULL},
    {0x2A000000000ULL, 0x2B000000000ULL},
    {0x2E000000000ULL, 0x2F000000000ULL},
    {0x3B000000000ULL, 0x3C000000000ULL},
    {0x3F000000000ULL, 0x40000000000ULL},
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
  while (true) {
    void *p = mmap(0, kMapSize, PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
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
