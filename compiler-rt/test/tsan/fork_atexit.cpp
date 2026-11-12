// RUN: %clangxx_tsan -O1 %s -o %t && %env_tsan_opts=atexit_sleep_ms=50 %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

void foo() {
  fprintf(stderr, "CHILD ATEXIT\n");
}

void *worker(void *unused) {
  return 0;
}

int main() {
  pthread_t t;
  pthread_create(&t, NULL, worker, NULL);
  int pid = fork();
  if (pid == 0) {
    // child
    atexit(foo);
    fprintf(stderr, "CHILD DONE\n");
  } else {
    pthread_join(t, 0);
    if (waitpid(pid, 0, 0) == -1) {
      perror("waitpid");
      exit(1);
    }
    fprintf(stderr, "PARENT DONE\n");
  }
}

// CHECK: CHILD DONE
// CHECK: CHILD ATEXIT
// CHECK: PARENT DONE
