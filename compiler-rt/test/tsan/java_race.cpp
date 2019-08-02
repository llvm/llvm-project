// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "java.h"

void *Thread(void *p) {
  barrier_wait(&barrier);
  *(int*)p = 42;
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  int const kHeapSize = 1024 * 1024;
  jptr jheap = (jptr)malloc(kHeapSize + 8) + 8;
  __tsan_java_init(jheap, kHeapSize);
  const int kBlockSize = 16;
  __tsan_java_alloc(jheap, kBlockSize);
  pthread_t th;
  pthread_create(&th, 0, Thread, (void*)jheap);
  *(int*)jheap = 43;
  barrier_wait(&barrier);
  pthread_join(th, 0);
  __tsan_java_free(jheap, kBlockSize);
  fprintf(stderr, "DONE\n");
  return __tsan_java_fini();
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: DONE
