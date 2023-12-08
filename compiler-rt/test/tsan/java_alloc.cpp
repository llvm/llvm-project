// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "java.h"

int const kHeapSize = 1024 * 1024;

void stress(jptr addr) {
  for (jptr sz = 8; sz <= 32; sz <<= 1) {
    for (jptr i = 0; i < kHeapSize / 4 / sz; i++) {
      __tsan_java_alloc(addr + i * sz, sz);
    }
    __tsan_java_move(addr, addr + kHeapSize / 2, kHeapSize / 4);
    __tsan_java_free(addr + kHeapSize / 2, kHeapSize / 4);
  }
}

void *Thread(void *p) {
  stress((jptr)p);
  return 0;
}

int main() {
  jptr jheap = (jptr)malloc(kHeapSize + 8) + 8;
  __tsan_java_init(jheap, kHeapSize);
  pthread_t th;
  pthread_create(&th, 0, Thread, (void*)(jheap + kHeapSize / 4));
  stress(jheap);
  pthread_join(th, 0);
  if (__tsan_java_fini() != 0) {
    fprintf(stderr, "FAILED\n");
    return 1;
  }
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK-NOT: FAILED
// CHECK: DONE
