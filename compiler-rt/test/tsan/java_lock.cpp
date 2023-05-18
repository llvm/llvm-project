// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "java.h"

jptr varaddr;
jptr lockaddr;

void *Thread(void *p) {
  barrier_wait(&barrier);
  __tsan_java_mutex_lock(lockaddr);
  *(int*)varaddr = 42;
  __tsan_java_mutex_unlock(lockaddr);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  int const kHeapSize = 1024 * 1024;
  jptr jheap = (jptr)malloc(kHeapSize + 8) + 8;
  __tsan_java_init(jheap, kHeapSize);
  const int kBlockSize = 16;
  __tsan_java_alloc(jheap, kBlockSize);
  varaddr = jheap;
  lockaddr = jheap + 8;
  pthread_t th;
  pthread_create(&th, 0, Thread, 0);
  __tsan_java_mutex_lock(lockaddr);
  *(int*)varaddr = 43;
  __tsan_java_mutex_unlock(lockaddr);
  barrier_wait(&barrier);
  pthread_join(th, 0);
  __tsan_java_free(jheap, kBlockSize);
  fprintf(stderr, "DONE\n");
  return __tsan_java_fini();
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: DONE
