// RUN: %clangxx_tsan %s -o %t && %run %t 2>&1 | FileCheck %s

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>

volatile bool signal_delivered;

static void handler(int sig) {
  if (sig == SIGALRM)
    signal_delivered = true;
}

static void* thr(void *p) {
  sigset_t sigset;
  sigemptyset(&sigset);
  sigaddset(&sigset, SIGALRM);
  int ret = pthread_sigmask(SIG_UNBLOCK, &sigset, NULL);
  if (ret) abort();

  struct sigaction act = {};
  act.sa_handler = &handler;
  if (sigaction(SIGALRM, &act, 0)) {
    perror("sigaction");
    exit(1);
  }

  itimerval t;
  t.it_value.tv_sec = 0;
  t.it_value.tv_usec = 10000;
  t.it_interval = t.it_value;
  if (setitimer(ITIMER_REAL, &t, 0)) {
    perror("setitimer");
    exit(1);
  }

  while (!signal_delivered) {
    usleep(1000);
  }

  t.it_value.tv_usec = 0;
  if (setitimer(ITIMER_REAL, &t, 0)) {
    perror("setitimer");
    exit(1);
  }

  fprintf(stderr, "SIGNAL DELIVERED\n");

  return 0;
}

int main() {
  sigset_t sigset;
  sigemptyset(&sigset);
  sigaddset(&sigset, SIGALRM);
  int ret = pthread_sigmask(SIG_BLOCK, &sigset, NULL);
  if (ret) abort();

  pthread_t th;
  pthread_create(&th, 0, thr, 0);
  pthread_join(th, 0);

  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: SIGNAL DELIVERED
// CHECK: DONE
// CHECK-NOT: WARNING: ThreadSanitizer:
