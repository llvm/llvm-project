// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "java.h"

jptr varaddr;
jptr lockaddr;
jptr varaddr2;
jptr lockaddr2;

void *Thread(void *p) {
  barrier_wait(&barrier);
  __tsan_java_mutex_lock(lockaddr2);
  *(int*)varaddr2 = 42;
  __tsan_java_mutex_unlock(lockaddr2);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  int const kHeapSize = 1024 * 1024;
  jptr jheap = (jptr)malloc(kHeapSize + 8) + 8;
  __tsan_java_init(jheap, kHeapSize);
  const int kBlockSize = 64;
  int const kMove = 1024;
  __tsan_java_alloc(jheap, kBlockSize);
  varaddr = jheap;
  lockaddr = jheap + 46;
  varaddr2 = varaddr + kMove;
  lockaddr2 = lockaddr + kMove;
  pthread_t th;
  pthread_create(&th, 0, Thread, 0);
  __tsan_java_mutex_lock(lockaddr);
  *(int*)varaddr = 43;
  __tsan_java_mutex_unlock(lockaddr);
  __tsan_java_move(varaddr, varaddr2, kBlockSize);
  barrier_wait(&barrier);
  pthread_join(th, 0);
  __tsan_java_free(varaddr2, kBlockSize);
  fprintf(stderr, "DONE\n");
  return __tsan_java_fini();
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: DONE
