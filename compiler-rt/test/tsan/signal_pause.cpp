// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

// Test that pause loop handles signals.

#include "test.h"
#include <signal.h>
#include <errno.h>

void handler(int signum) {
  write(2, "DONE\n", 5);
  _exit(0);
}

void *thread(void *arg) {
  for (;;)
    pause();
  return 0;
}

int main(int argc, char** argv) {
  struct sigaction act = {};
  act.sa_handler = &handler;
  if (sigaction(SIGUSR1, &act, 0)) {
    fprintf(stderr, "sigaction failed %d\n", errno);
    return 1;
  }
  pthread_t th;
  pthread_create(&th, 0, thread, 0);
  sleep(1);  // give it time to block in pause
  pthread_kill(th, SIGUSR1);
  sleep(10);  // signal handler must exit the process while we are here
  return 0;
}

// CHECK: DONE
