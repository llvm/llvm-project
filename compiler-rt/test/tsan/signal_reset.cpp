// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: darwin
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>

volatile int X;
int stop;

static void handler(int sig) {
  (void)sig;
  if (X != 0)
    printf("bad");
}

static void* busy(void *p) {
  while (__atomic_load_n(&stop, __ATOMIC_RELAXED) == 0) {
  }
  return 0;
}

static void* reset(void *p) {
  struct sigaction act = {};
  for (int i = 0; i < 1000000; i++) {
    act.sa_handler = &handler;
    if (sigaction(SIGALRM, &act, 0)) {
      perror("sigaction");
      exit(1);
    }
    act.sa_handler = SIG_IGN;
    if (sigaction(SIGALRM, &act, 0)) {
      perror("sigaction");
      exit(1);
    }
  }
  return 0;
}

int main() {
  struct sigaction act = {};
  act.sa_handler = SIG_IGN;
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

  pthread_t th[2];
  pthread_create(&th[0], 0, busy, 0);
  pthread_create(&th[1], 0, reset, 0);

  pthread_join(th[1], 0);
  __atomic_store_n(&stop, 1, __ATOMIC_RELAXED);
  pthread_join(th[0], 0);

  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: DONE
// CHECK-NOT: WARNING: ThreadSanitizer:
