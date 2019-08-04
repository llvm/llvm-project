// RUN: %clangxx_tsan -O1 -DTEST_ERROR=ERANGE %s -o %t && %run %t 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-SYS %s
// RUN: %clangxx_tsan -O1 -DTEST_ERROR=-1 %s -o %t && not %run %t 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-USER %s
// This test is for GNU specific version of strerror_r()
// UNSUPPORTED: darwin, netbsd, freebsd

#include "test.h"

#include <errno.h>
#include <pthread.h>
#include <string.h>

char buffer[1000];

void *Thread(void *p) {
  barrier_wait(&barrier);
  return strerror_r(TEST_ERROR, buffer, sizeof(buffer));
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t th;
  pthread_create(&th, 0, Thread, 0);
  strerror_r(TEST_ERROR, buffer, sizeof(buffer));
  barrier_wait(&barrier);
  pthread_join(th, 0);
  fprintf(stderr, "DONE\n");
}

// CHECK-USER: WARNING: ThreadSanitizer: data race
// CHECK-SYS-NOT: WARNING: ThreadSanitizer: data race

// CHECK: DONE
