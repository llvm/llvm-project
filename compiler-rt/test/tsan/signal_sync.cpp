// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: darwin
#include "test.h"
#include <signal.h>
#include <sys/types.h>
#include <sys/time.h>
#include <errno.h>

volatile int X;

static void handler(int sig) {
  (void)sig;
  if (X != 42)
    printf("bad");
}

static void* thr(void *p) {
  for (int i = 0; i != 1000; i++)
    usleep(1000);  // process signals
  return 0;
}

int main() {
  const int kThreads = 10;
  pthread_t th[kThreads];
  for (int i = 0; i < kThreads; i++)
    pthread_create(&th[i], 0, thr, 0);

  X = 42;

  struct sigaction act = {};
  act.sa_handler = &handler;
  if (sigaction(SIGALRM, &act, 0)) {
    perror("sigaction");
    exit(1);
  }

  itimerval t;
  t.it_value.tv_sec = 0;
  t.it_value.tv_usec = 10;
  t.it_interval = t.it_value;
  if (setitimer(ITIMER_REAL, &t, 0)) {
    perror("setitimer");
    exit(1);
  }

  for (int i = 0; i < kThreads; i++)
    pthread_join(th[i], 0);

  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: DONE
// CHECK-NOT: WARNING: ThreadSanitizer:
