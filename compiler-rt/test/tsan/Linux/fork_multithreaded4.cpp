// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

// The test tries to provoke internal allocator to be locked during fork
// and then force the child process to use the internal allocator.

#include "../test.h"
#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>

static void *forker(void *arg) {
  void *p = calloc(1, 16);
  static_cast<volatile int *>(p)[0]++;
  __atomic_fetch_add(static_cast<int *>(p), 1, __ATOMIC_SEQ_CST);
  int pid = fork();
  if (pid < 0) {
    fprintf(stderr, "failed to fork (%d)\n", errno);
    exit(1);
  }
  if (pid == 0) {
    __atomic_fetch_add(&static_cast<int *>(p)[1], 1, __ATOMIC_SEQ_CST);
    exit(0);
  }
  int status = 0;
  while (waitpid(pid, &status, 0) != pid) {
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    fprintf(stderr, "subprocess failed (%d)\n", status);
    exit(1);
  }
  free(p);
  return 0;
}

int main() {
  for (int i = 0; i < 10; i++) {
    pthread_t threads[100];
    for (auto &th : threads)
      pthread_create(&th, 0, forker, 0);
    for (auto th : threads)
      pthread_join(th, 0);
  }
  fprintf(stderr, "DONE\n");
}

// CHECK: DONE
