// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t 2>&1 | FileCheck %s
#include "test.h"
#include <stdint.h>
#include <errno.h>
#include <sys/mman.h>

// Test for issue:
// https://github.com/google/sanitizers/issues/412

// MAP_32BIT flag for mmap is supported only for x86_64.
// XFAIL: mips
// XFAIL: aarch64
// XFAIL: powerpc64
// XFAIL: s390x

// MAP_32BIT doesn't exist on OS X and NetBSD.
// UNSUPPORTED: darwin,netbsd

void *Thread(void *ptr) {
  *(int*)ptr = 42;
  barrier_wait(&barrier);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  void *ptr = mmap(0, 128 << 10, PROT_READ|PROT_WRITE,
      MAP_32BIT|MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  fprintf(stderr, "ptr=%p\n", ptr);
  if (ptr == MAP_FAILED) {
    fprintf(stderr, "mmap failed: %d\n", errno);
    return 1;
  }
  if ((uintptr_t)ptr >= (1ull << 32)) {
    fprintf(stderr, "ptr is too high\n");
    return 1;
  }
  pthread_t t;
  pthread_create(&t, 0, Thread, ptr);
  barrier_wait(&barrier);
  *(int*)ptr = 42;
  pthread_join(t, 0);
  munmap(ptr, 128 << 10);
  fprintf(stderr, "DONE\n");
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: DONE
