// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

// Test that a signal is not delivered when it is blocked.

#include "test.h"
#include <semaphore.h>
#include <signal.h>
#include <errno.h>

int stop;
sig_atomic_t signal_blocked;

void handler(int signum) {
  if (signal_blocked) {
    fprintf(stderr, "signal arrived when blocked\n");
    exit(1);
  }
}

void *thread(void *arg) {
  sigset_t myset;
  sigemptyset(&myset);
  sigaddset(&myset, SIGUSR1);
  while (!__atomic_load_n(&stop, __ATOMIC_RELAXED)) {
    usleep(1);
    if (pthread_sigmask(SIG_BLOCK, &myset, 0)) {
      fprintf(stderr, "pthread_sigmask failed %d\n", errno);
      exit(1);
    }
    signal_blocked = 1;
    usleep(1);
    signal_blocked = 0;
    if (pthread_sigmask(SIG_UNBLOCK, &myset, 0)) {
      fprintf(stderr, "pthread_sigmask failed %d\n", errno);
      exit(1);
    }
  }
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
  for (int i = 0; i < 100000; i++)
    pthread_kill(th, SIGUSR1);
  __atomic_store_n(&stop, 1, __ATOMIC_RELAXED);
  pthread_join(th, 0);
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: ThreadSanitizer CHECK
// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: DONE
