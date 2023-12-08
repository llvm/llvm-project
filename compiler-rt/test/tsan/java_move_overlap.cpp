// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %run %t arg 2>&1 | FileCheck %s
#include "java.h"

jptr varaddr1_old;
jptr varaddr2_old;
jptr lockaddr1_old;
jptr lockaddr2_old;
jptr varaddr1_new;
jptr varaddr2_new;
jptr lockaddr1_new;
jptr lockaddr2_new;

void *Thread(void *p) {
  barrier_wait(&barrier);
  __tsan_java_mutex_lock(lockaddr1_new);
  *(char*)varaddr1_new = 43;
  __tsan_java_mutex_unlock(lockaddr1_new);
  __tsan_java_mutex_lock(lockaddr2_new);
  *(char*)varaddr2_new = 43;
  __tsan_java_mutex_unlock(lockaddr2_new);
  return 0;
}

int main(int argc, char **argv) {
  barrier_init(&barrier, 2);
  int const kHeapSize = 1024 * 1024;
  void *jheap = malloc(kHeapSize);
  jheap = (char*)jheap + 8;
  __tsan_java_init((jptr)jheap, kHeapSize);
  const int kBlockSize = 64;
  int const kMove = 32;
  varaddr1_old = (jptr)jheap;
  lockaddr1_old = (jptr)jheap + 1;
  varaddr2_old = (jptr)jheap + kBlockSize - 1;
  lockaddr2_old = (jptr)jheap + kBlockSize - 16;
  varaddr1_new = varaddr1_old + kMove;
  lockaddr1_new = lockaddr1_old + kMove;
  varaddr2_new = varaddr2_old + kMove;
  lockaddr2_new = lockaddr2_old + kMove;
  if (argc > 1) {
    // Move memory backwards.
    varaddr1_old += kMove;
    lockaddr1_old += kMove;
    varaddr2_old += kMove;
    lockaddr2_old += kMove;
    varaddr1_new -= kMove;
    lockaddr1_new -= kMove;
    varaddr2_new -= kMove;
    lockaddr2_new -= kMove;
  }
  __tsan_java_alloc(varaddr1_old, kBlockSize);

  pthread_t th;
  pthread_create(&th, 0, Thread, 0);

  __tsan_java_mutex_lock(lockaddr1_old);
  *(char*)varaddr1_old = 43;
  __tsan_java_mutex_unlock(lockaddr1_old);
  __tsan_java_mutex_lock(lockaddr2_old);
  *(char*)varaddr2_old = 43;
  __tsan_java_mutex_unlock(lockaddr2_old);

  __tsan_java_move(varaddr1_old, varaddr1_new, kBlockSize);
  barrier_wait(&barrier);
  pthread_join(th, 0);
  __tsan_java_free(varaddr1_new, kBlockSize);
  fprintf(stderr, "DONE\n");
  return __tsan_java_fini();
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: DONE
