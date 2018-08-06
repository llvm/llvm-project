// Test forked process does not run lsan.
// RUN: %clangxx_lsan %s -o %t && %run %t 2>&1 | FileCheck %s

#include <pthread.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

static pthread_barrier_t barrier;

// CHECK-NOT: SUMMARY: {{(Leak|Address)}}Sanitizer:
static void *thread_func(void *arg) {
  void *buffer = malloc(1337);
  pthread_barrier_wait(&barrier);
  for (;;)
    pthread_yield();
  return 0;
}

int main() {
  pthread_barrier_init(&barrier, 0, 2);
  pthread_t tid;
  int res = pthread_create(&tid, 0, thread_func, 0);
  pthread_barrier_wait(&barrier);
  pthread_barrier_destroy(&barrier);

  pid_t pid = fork();
  if (pid > 0) {
    int status = 0;
    waitpid(pid, &status, 0);
  }
  return 0;
}

// CHECK: WARNING: LeakSanitizer is disabled in forked process
