// RUN: %clangxx_tsan -O1 -DTEST_ERROR=ERANGE %s -o %t && %run %t 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-SYS %s
// RUN: %clangxx_tsan -O1 -DTEST_ERROR=-1 %s -o %t && not %run %t 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-USER %s
// UNSUPPORTED: darwin

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

char buffer[1000];

void *Thread(void *p) {
  return strerror_r(TEST_ERROR, buffer, sizeof(buffer));
}

int main() {
  pthread_t th[2];
  pthread_create(&th[0], 0, Thread, 0);
  pthread_create(&th[1], 0, Thread, 0);
  pthread_join(th[0], 0);
  pthread_join(th[1], 0);
  fprintf(stderr, "DONE\n");
}

// CHECK-USER: WARNING: ThreadSanitizer: data race
// CHECK-SYS-NOT: WARNING: ThreadSanitizer: data race

// CHECK: DONE
