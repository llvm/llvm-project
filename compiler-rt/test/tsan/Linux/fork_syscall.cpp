// RUN: %clangxx_tsan -O1 %s -o %t && %env_tsan_opts=atexit_sleep_ms=50 %run %t 2>&1 | FileCheck %s
#include "../test.h"
#include "syscall.h"
#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>

int counter;

static void *incrementer(void *p) {
  for (;;)
    __sync_fetch_and_add(&counter, 1);
  return 0;
}

int main() {
  pthread_t th1;
  pthread_create(&th1, 0, incrementer, 0);
  for (int i = 0; i < 10; i++) {
    switch (myfork()) {
    default: // parent
      while (wait(0) < 0) {
      }
      fprintf(stderr, ".");
      break;
    case 0: // child
      __sync_fetch_and_add(&counter, 1);
      exit(0);
      break;
    case -1: // error
      fprintf(stderr, "failed to fork (%d)\n", errno);
      exit(1);
    }
  }
  fprintf(stderr, "OK\n");
}

// CHECK: OK
