// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s -check-prefix=CHECK-DIE
// RUN: %clangxx_tsan -O1 %s -o %t && %env_tsan_opts=die_after_fork=0 %run %t 2>&1 | FileCheck %s -check-prefix=CHECK-NODIE
#include "test.h"
#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>

static void *sleeper(void *p) {
  sleep(1000);  // not intended to exit during test
  return 0;
}

static void *nop(void *p) {
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t th;
  pthread_create(&th, 0, sleeper, 0);
  switch (fork()) {
  default:  // parent
    while (wait(0) < 0) {}
    break;
  case 0:  // child
    {
      pthread_t th2;
      pthread_create(&th2, 0, nop, 0);
      exit(0);
      break;
    }
  case -1:  // error
    fprintf(stderr, "failed to fork (%d)\n", errno);
    exit(1);
  }
  fprintf(stderr, "OK\n");
}

// CHECK-DIE: ThreadSanitizer: starting new threads after multi-threaded fork is not supported
// CHECK-DIE: OK

// CHECK-NODIE-NOT: ThreadSanitizer: starting new threads after multi-threaded fork is not supported
// CHECK-NODIE: OK

