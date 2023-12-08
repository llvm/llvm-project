// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"

/*
Annotations usage example.

Tsan does not see synchronization in barrier_wait.
ANNOTATE_HAPPENS_BEFORE/AFTER communicate the synchronization to tsan
and prevent the race report.
*/

int Global;

void *Thread1(void *x) {
  barrier_wait(&barrier);
  ANNOTATE_HAPPENS_AFTER(&barrier);
  Global++;
  return NULL;
}

void *Thread2(void *x) {
  Global--;
  ANNOTATE_HAPPENS_BEFORE(&barrier);
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
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: DONE

