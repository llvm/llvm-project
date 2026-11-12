// RUN: %clang_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
// Regression test for
// https://groups.google.com/d/msg/thread-sanitizer/e_zB9gYqFHM/DmAiTsrLAwAJ
// pthread_atfork() callback triggers a data race and we deadlocked
// on the report_mtx as we lock it around fork.
#include "test.h"
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>

int glob = 0;

void *worker(void *unused) {
  glob++;
  barrier_wait(&barrier);
  return NULL;
}

void atfork() {
  glob++;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_atfork(atfork, atfork, atfork);
  pthread_t t;
  pthread_create(&t, NULL, worker, NULL);
  barrier_wait(&barrier);
  pid_t pid = fork();
  if (pid < 0) {
    fprintf(stderr, "fork failed: %d\n", errno);
    return 1;
  }
  if (pid == 0) {
    fprintf(stderr, "CHILD\n");
    return 0;
  }
  if (pid != waitpid(pid, NULL, 0)) {
    fprintf(stderr, "waitpid failed: %d\n", errno);
    return 1;
  }
  pthread_join(t, NULL);
  fprintf(stderr, "PARENT\n");
  return 0;
}

// CHECK: ThreadSanitizer: data race
// CHECK:   Write of size 4
// CHECK:     #0 atfork
// CHECK:   Previous write of size 4
// CHECK:     #0 worker
// CHECK: CHILD
// CHECK: PARENT
