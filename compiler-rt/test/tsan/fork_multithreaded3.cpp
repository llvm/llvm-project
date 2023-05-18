// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

static void *racer(void *p) {
  *(int*)p = 42;
  return 0;
}

int main() {
  switch (fork()) {
  default:  // parent
    while (wait(0) < 0) {}
    break;
  case 0:  // child
    {
      int x = 0;
      pthread_t th1, th2;
      pthread_create(&th1, 0, racer, &x);
      pthread_create(&th2, 0, racer, &x);
      pthread_join(th1, 0);
      pthread_join(th2, 0);
      exit(0);
      break;
    }
  case -1:  // error
    fprintf(stderr, "failed to fork (%d)\n", errno);
    exit(1);
  }
  fprintf(stderr, "OK\n");
}

// CHECK: ThreadSanitizer: data race
// CHECK: OK

