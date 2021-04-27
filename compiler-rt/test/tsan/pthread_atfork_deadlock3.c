// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// Regression test for
// https://groups.google.com/g/thread-sanitizer/c/TQrr4-9PRYo/m/HFR4FMi6AQAJ
#include "test.h"
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>
#include <string.h>
#include <signal.h>

long glob = 0;

void *worker(void *main) {
  glob++;
  barrier_wait(&barrier);
  barrier_wait(&barrier);
  pthread_kill((pthread_t)main, SIGPROF);
  barrier_wait(&barrier);
  return NULL;
}

void atfork() {
  barrier_wait(&barrier);
  barrier_wait(&barrier);
  write(2, "in atfork\n", strlen("in atfork\n"));
  static volatile long a;
  __atomic_fetch_add(&a, 1, __ATOMIC_RELEASE);
}

void handler(int sig) {
  write(2, "in handler\n", strlen("in handler\n"));
  glob++;
}

int main() {
  barrier_init(&barrier, 2);
  struct sigaction act = {};
  act.sa_handler = &handler;
  if (sigaction(SIGPROF, &act, 0)) {
    perror("sigaction");
    exit(1);
  }
  pthread_atfork(atfork, NULL, NULL);
  pthread_t t;
  pthread_create(&t, NULL, worker, (void*)pthread_self());
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

// CHECK: in atfork
// CHECK: in handler
// Note: There is a race, but we won't report it
// to not deadlock.
// CHECK-NOT: ThreadSanitizer: data race
// CHECK: CHILD
// CHECK: PARENT
