// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

// UNSUPPORTED: ios

#include <stdint.h>
#include <stdio.h>
#include <errno.h>
#include <sys/mman.h>

#if defined(__FreeBSD__)
// The MAP_NORESERVE define has been removed in FreeBSD 11.x, and even before
// that, it was never implemented.  So just define it to zero.
#undef  MAP_NORESERVE
#define MAP_NORESERVE 0
#endif

int main() {
#ifdef __x86_64__
  const size_t kLog2Size = 39;
#elif defined(__mips64) || defined(__aarch64__) || defined(__loongarch_lp64)
  const size_t kLog2Size = 32;
#elif defined(__powerpc64__)
  const size_t kLog2Size = 39;
#elif defined(__s390x__)
  const size_t kLog2Size = 43;
#endif
  const uintptr_t kLocation = 0x40ULL << kLog2Size;
  void *p = mmap(
      reinterpret_cast<void*>(kLocation),
      1ULL << kLog2Size,
      PROT_READ|PROT_WRITE,
      MAP_PRIVATE|MAP_ANON|MAP_NORESERVE,
      -1, 0);
  fprintf(stderr, "DONE %p %d\n", p, errno);
  return p == MAP_FAILED;
}

// CHECK: DONE
