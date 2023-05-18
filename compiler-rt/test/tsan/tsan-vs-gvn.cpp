// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_tsan -O2 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_tsan -O3 %s -o %t && %run %t 2>&1 | FileCheck %s
//
// Check that load widening is not tsan-hostile.
#include <pthread.h>
#include <stdio.h>
#include <string.h>

struct {
  int i;
  char c1, c2, c3, c4;
} S;

int G;

void *Thread1(void *x) {
  G = S.c1 + S.c3;
  return NULL;
}

void *Thread2(void *x) {
  S.c2 = 1;
  return NULL;
}

int main() {
  pthread_t t[2];
  memset(&S, 123, sizeof(S));
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  fprintf(stderr, "PASS\n");
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: PASS
