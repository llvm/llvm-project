// RUN: %clangxx_tsan %s -o %t && %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: darwin

// It's very flaky on PPC with COMPILER_RT_DEBUG.
// UNSUPPORTED: !compiler-rt-optimized && ppc

// Test case for https://github.com/google/sanitizers/issues/1540

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

volatile int X;

static void handler(int sig) {
  (void)sig;
  if (X != 0)
    printf("bad");
}

static void *thr1(void *p) {
  sleep(1);
  return 0;
}

static void *thr(void *p) {
  pthread_t th[10];
  for (int i = 0; i < sizeof(th) / sizeof(th[0]); i++)
    pthread_create(&th[i], 0, thr1, 0);
  for (int i = 0; i < sizeof(th) / sizeof(th[0]); i++)
    pthread_join(th[i], 0);
  return 0;
}

int main() {
  struct sigaction act = {};
  act.sa_handler = &handler;
  if (sigaction(SIGPROF, &act, 0)) {
    perror("sigaction");
    exit(1);
  }

  itimerval t;
  t.it_value.tv_sec = 0;
  t.it_value.tv_usec = 10;
  t.it_interval = t.it_value;
  if (setitimer(ITIMER_PROF, &t, 0)) {
    perror("setitimer");
    exit(1);
  }

  pthread_t th[100];
  for (int i = 0; i < sizeof(th) / sizeof(th[0]); i++)
    pthread_create(&th[i], 0, thr, 0);
  for (int i = 0; i < sizeof(th) / sizeof(th[0]); i++)
    pthread_join(th[i], 0);

  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: DONE
// CHECK-NOT: WARNING: ThreadSanitizer:
