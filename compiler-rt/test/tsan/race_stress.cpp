// RUN: %clangxx_tsan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
#include "test.h"

const int kThreads = 16;
const int kIters = 1000;

volatile int X = 0;

void *thr(void *arg) {
  for (int i = 0; i < kIters; i++)
    X++;
  return 0;
}

int main() {
  pthread_t th[kThreads];
  for (int i = 0; i < kThreads; i++)
    pthread_create(&th[i], 0, thr, 0);
  for (int i = 0; i < kThreads; i++)
    pthread_join(th[i], 0);
  fprintf(stderr, "DONE\n");
}

// CHECK: ThreadSanitizer: data race
// CHECK: DONE
