// RUN: %clangxx -O0 %s -o %t && %env_tool_opts=verbosity=10 %run %t 2>&1 | FileCheck %s

#include <pthread.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>

void *thread(void *unused) {
  printf("PID: %d\n", getpid());
  printf("TID: %ld\n", syscall(SYS_gettid));
  fflush(stdout);
  return 0;
}

int main() {
  pthread_t t;
  pthread_create(&t, 0, thread, 0);
  pthread_join(t, 0);
  return 0;
}
// CHECK: PID: [[PID:[0-9]+]]
// CHECK: TID: [[TID:[0-9]+]]
// CHECK: ==[[PID]]:[[TID]]==
