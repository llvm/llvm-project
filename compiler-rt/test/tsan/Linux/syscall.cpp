// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t 2>&1 | FileCheck %s
#include "syscall.h"
#include "../test.h"
#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>

int pipefd[2];
char buf[10];

static void *thr(void *p) {
  barrier_wait(&barrier);
  mywrite(pipefd[1], buf, sizeof(buf));
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  if (mypipe(pipefd))
    exit((perror("pipe"), 1));
  mywrite(pipefd[1], buf, sizeof(buf));
  pthread_t th;
  pthread_create(&th, 0, thr, 0);
  myread(pipefd[0], buf, sizeof(buf));
  barrier_wait(&barrier);
  pthread_join(th, 0);
  fprintf(stderr, "DONE\n");
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Read of size 8
// CHECK:     #0 mywrite
// CHECK:     #1 thr
// CHECK:   Previous write of size 8
// CHECK:     #0 myread
// CHECK:     #1 main
// CHECK: DONE
