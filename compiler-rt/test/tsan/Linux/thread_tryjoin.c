// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#define _GNU_SOURCE
#include "../test.h"
#include <errno.h>

int var;

void *Thread(void *x) {
  barrier_wait(&barrier);
  var = 1;
  return 0;
}

static void check(int res) {
  if (res != EBUSY) {
    fprintf(stderr, "Unexpected result of pthread_tryjoin_np: %d\n", res);
    exit(1);
  }
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  check(pthread_tryjoin_np(t, 0));
  barrier_wait(&barrier);
  for (;;) {
    int res = pthread_tryjoin_np(t, 0);
    if (!res)
      break;
    check(res);
    pthread_yield();
  }
  var = 2;
  fprintf(stderr, "PASS\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK-NOT: WARNING: ThreadSanitizer: thread leak
// CHECK: PASS
