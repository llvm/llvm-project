// RUN: echo "race_top:TopFunction" > %t.supp
// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %env_tsan_opts=suppressions='%t.supp' %run %t 2>&1 | FileCheck %s
// RUN: rm %t.supp
#include "test.h"

int Global;

void TopFunction(int *p) {
  *p = 1;
}

void *Thread(void *x) {
  barrier_wait(&barrier);
  TopFunction(&Global);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  Global--;
  barrier_wait(&barrier);
  pthread_join(t, 0);
  fprintf(stderr, "DONE\n");
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
