// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// REQUIRES: glibc-2.30
#include <pthread.h>
#include <stdio.h>

pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
struct timespec ts = {0};

void *tfunc(void *p) {
  if (!pthread_mutex_trylock(&m)) {
    puts("Second thread could not lock mutex");
    pthread_mutex_unlock(&m);
  }
  return p;
}

int main() {
  if (!pthread_mutex_clocklock(&m, CLOCK_REALTIME, &ts)) {
    pthread_t thr;
    pthread_create(&thr, 0, tfunc, 0);
    pthread_join(thr, 0);
    pthread_mutex_unlock(&m);
  } else
    puts("Failed to lock mutex");
  fprintf(stderr, "PASS\n");
}

// CHECK-NOT: WARNING: ThreadSanitizer: unlock of an unlocked mutex
// CHECK: PASS
