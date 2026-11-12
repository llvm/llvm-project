// RUN: %clang_tsan -O1 %s -o %t && %env_tsan_opts=suppressions='%s.supp' %run %t 2>&1 | FileCheck %s
#include "test.h"

int Global;

void *Thread1(void *x) {
  barrier_wait(&barrier);
  Global = 42;
  return NULL;
}

void *Thread2(void *x) {
  Global = 43;
  barrier_wait(&barrier);
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  fprintf(stderr, "OK\n");
  return 0;
}

// CHECK-NOT: failed to open suppressions file
// CHECK-NOT: WARNING: ThreadSanitizer: data race

