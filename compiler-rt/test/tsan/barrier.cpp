// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// CHECK-NOT: ThreadSanitizer: data race
// CHECK: DONE

// pthread barriers are not available on OS X
// UNSUPPORTED: darwin

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

const int kSize = 4;
volatile int kIter = 10;  // prevent unwinding
int data[2][kSize];
pthread_barrier_t barrier;

void *thr(void *p) {
  int idx = (int)(long)p;
  for (int i = 0; i < kIter; i++) {
    int *prev = data[i % 2];
    int *curr = data[(i + 1) % 2];
    int left = idx - 1 >= 0 ? prev[idx - 1] : 0;
    int right = idx + 1 < kSize ? prev[idx + 1] : 0;
    curr[idx] = (left + right) / 2;
    pthread_barrier_wait(&barrier);
  }
  return 0;
}

int main() {
  pthread_barrier_init(&barrier, 0, kSize);
  pthread_t th[kSize];
  for (int i = 0; i < kSize; i++)
    pthread_create(&th[i], 0, thr, (void*)(long)i);
  for (int i = 0; i < kSize; i++)
    pthread_join(th[i], 0);
  pthread_barrier_destroy(&barrier);
  fprintf(stderr, "DONE\n");
}
