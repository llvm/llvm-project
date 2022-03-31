// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 66 2>&1 | FileCheck %s -check-prefix=CHECK-DIE
// RUN: %clangxx_tsan -O1 %s -o %t && %env_tsan_opts=die_after_fork=0 %run %t 2>&1 | FileCheck %s -check-prefix=CHECK-NODIE
#include "test.h"
#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>

static void *sleeper(void *p) {
  barrier_wait(&barrier);
  return 0;
}

static void *nop(void *p) {
  return 0;
}

int main(int argc, const char **argv) {
  barrier_init(&barrier, 3);
  const int kSleeperThreads = 2;
  barrier_init(&barrier, kSleeperThreads + 1);
  pthread_t th0[kSleeperThreads];
  for (int i = 0; i < kSleeperThreads; i++)
    pthread_create(&th0[i], 0, sleeper, 0);
  const int kNopThreads = 5;
  pthread_t th1[kNopThreads];
  for (int i = 0; i < kNopThreads; i++)
    pthread_create(&th1[i], 0, nop, 0);
  for (int i = 0; i < kNopThreads; i++)
    pthread_join(th1[i], 0);
  int pid = fork();
  if (pid < 0) {
    fprintf(stderr, "failed to fork (%d)\n", errno);
    exit(1);
  }
  if (pid == 0) {
    // child
    const int kChildThreads = 4;
    pthread_t th2[kChildThreads];
    for (int i = 0; i < kChildThreads; i++)
      pthread_create(&th2[i], 0, nop, 0);
    for (int i = 0; i < kChildThreads; i++)
      pthread_join(th2[i], 0);
    exit(0);
    return 0;
  }
  // parent
  int expect = argc > 1 ? atoi(argv[1]) : 0;
  int status = 0;
  while (waitpid(pid, &status, 0) != pid) {
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != expect) {
    fprintf(stderr, "subprocess exited with %d, expected %d\n", status, expect);
    exit(1);
  }
  barrier_wait(&barrier);
  for (int i = 0; i < kSleeperThreads; i++)
    pthread_join(th0[i], 0);
  fprintf(stderr, "OK\n");
  return 0;
}

// CHECK-DIE: ThreadSanitizer: starting new threads after multi-threaded fork is not supported

// CHECK-NODIE-NOT: ThreadSanitizer:
// CHECK-NODIE: OK
// CHECK-NODIE-NOT: ThreadSanitizer:
