// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %deflake %run %t 2>&1 | FileCheck %s
// RUN: %deflake %run %t arg 2>&1 | FileCheck %s
#include "java.h"

jptr varaddr1_old;
jptr varaddr2_old;
jptr varaddr1_new;
jptr varaddr2_new;

void *Thread(void *p) {
  barrier_wait(&barrier);
  *(int*)varaddr1_new = 43;
  *(int*)varaddr2_new = 43;
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
  varaddr2_old = (jptr)jheap + kBlockSize - 1;
  varaddr1_new = varaddr1_old + kMove;
  varaddr2_new = varaddr2_old + kMove;
  if (argc > 1) {
    // Move memory backwards.
    varaddr1_old += kMove;
    varaddr2_old += kMove;
    varaddr1_new -= kMove;
    varaddr2_new -= kMove;
  }
  __tsan_java_alloc(varaddr1_old, kBlockSize);

  pthread_t th;
  pthread_create(&th, 0, Thread, 0);

  *(int*)varaddr1_old = 43;
  *(int*)varaddr2_old = 43;

  __tsan_java_move(varaddr1_old, varaddr1_new, kBlockSize);
  barrier_wait(&barrier);
  pthread_join(th, 0);
  __tsan_java_free(varaddr1_new, kBlockSize);
  fprintf(stderr, "DONE\n");
  return __tsan_java_fini();
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: DONE
