// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

#include "java.h"
#include <errno.h>
#include <sys/mman.h>

int main() {
  // Test a non-regular kHeapSize
  // Previously __tsan_java_init failed because it encountered non-zero meta
  // shadow for the destination.
  size_t const kPageSize = sysconf(_SC_PAGESIZE);
  int const kSize = kPageSize - 1;
  jptr jheap2 = (jptr)mmap(0, kSize, PROT_READ | PROT_WRITE,
                           MAP_ANON | MAP_PRIVATE, -1, 0);
  if (jheap2 == (jptr)MAP_FAILED)
    return printf("mmap failed with %d\n", errno);
  __atomic_store_n((int *)(jheap2 + kSize - 3), 1, __ATOMIC_RELEASE);
  // Due to the previous incorrect meta-end calculation, the following munmap
  // did not clear the tail meta shadow.
  munmap((void *)jheap2, kSize);
  int const kHeapSize2 = kSize + 1;
  jheap2 = (jptr)mmap((void *)jheap2, kHeapSize2, PROT_READ | PROT_WRITE,
                      MAP_ANON | MAP_PRIVATE, -1, 0);
  if (jheap2 == (jptr)MAP_FAILED)
    return printf("second mmap failed with %d\n", errno);
  __tsan_java_init(jheap2, kHeapSize2);
  __tsan_java_move(jheap2, jheap2 + kHeapSize2 - 8, 8);
  fprintf(stderr, "DONE\n");
  return __tsan_java_fini();
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: DONE
