// RUN: %clangxx_tsan -O1 %s -o %t && %env_tsan_opts=atexit_sleep_ms=0 %run %t 2>&1 | FileCheck %s
#include "../test.h"
#include <errno.h>
#include <sched.h>
#include <sys/types.h>
#include <sys/wait.h>

long counter;

static void *incrementer(void *arg) {
  for (;;)
    __sync_fetch_and_add(&counter, 1);
  return 0;
}

static int cloned(void *arg) {
  for (int i = 0; i < 1000; i++)
    __sync_fetch_and_add(&counter, 1);
  exit(0);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t th;
  pthread_create(&th, 0, incrementer, 0);
  for (int i = 0; i < 100; i++) {
    char stack[64 << 10] __attribute__((aligned(64)));
    int pid = clone(cloned, stack + sizeof(stack), SIGCHLD, 0);
    if (pid == -1) {
      fprintf(stderr, "failed to clone: %d\n", errno);
      exit(1);
    }
    while (wait(0) != pid) {
    }
  }
  fprintf(stderr, "DONE\n");
}

// CHECK: DONE
