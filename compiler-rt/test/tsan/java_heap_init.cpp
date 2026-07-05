// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "java.h"
#include <errno.h>
#include <sys/mman.h>

int main() {
  // Test that munmap interceptor resets meta shadow for the memory range.
  // Previously __tsan_java_move failed because it encountered non-zero meta
  // shadow for the destination.
  int const kHeapSize = 1024 * 1024;
  jptr jheap = (jptr)mmap(0, kHeapSize, PROT_READ | PROT_WRITE,
      MAP_ANON | MAP_PRIVATE, -1, 0);
  if (jheap == (jptr)MAP_FAILED)
    return printf("mmap failed with %d\n", errno);
  __atomic_store_n((int*)jheap, 1, __ATOMIC_RELEASE);
  munmap((void*)jheap, kHeapSize);
  jheap = (jptr)mmap((void*)jheap, kHeapSize, PROT_READ | PROT_WRITE,
      MAP_ANON | MAP_PRIVATE, -1, 0);
  if (jheap == (jptr)MAP_FAILED)
    return printf("second mmap failed with %d\n", errno);
  __tsan_java_init(jheap, kHeapSize);
  __tsan_java_move(jheap + 16, jheap, 16);
  fprintf(stderr, "DONE\n");
  return __tsan_java_fini();
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: DONE
