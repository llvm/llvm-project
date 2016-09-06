// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>

int fds[2];
int X;

void *Thread1(void *x) {
  X = 42;
  write(fds[1], "a", 1);
  return NULL;
}

void *Thread2(void *x) {
  char buf;
  while (read(fds[0], &buf, 1) != 1) {
  }
  X = 43;
  return NULL;
}

int main() {
  pipe(fds);
  int pid = vfork();
  if (pid < 0) {
    fprintf(stderr, "FAIL to vfork\n");
    exit(1);
  }
  if (pid == 0) {  // child
    // Closing of fds must not affect parent process.
    // Strictly saying this is undefined behavior, because vfork child is not
    // allowed to call any functions other than exec/exit. But this is what
    // openjdk does.
    close(fds[0]);
    close(fds[1]);
    _exit(0);
  }
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  fprintf(stderr, "DONE\n");
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK-NOT: FAIL to vfork
// CHECK: DONE
