// RUN: %clang_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
// This test fails on powerpc64 BE (VMA=44), it does not appear to be
// a functional problem, but the Tsan report is missing some info.
// XFAIL: powerpc64-unknown-linux-gnu

#include "test.h"
#include <signal.h>
#include <sys/types.h>
#include <errno.h>

pthread_t mainth;
volatile int done;

static void MyHandler(int, siginfo_t *s, void *c) {
  errno = 1;
  done = 1;
}

static void* sendsignal(void *p) {
  barrier_wait(&barrier);
  pthread_kill(mainth, SIGPROF);
  return 0;
}

static __attribute__((noinline)) void loop() {
  barrier_wait(&barrier);
  while (done == 0) {
    volatile char *p = (char*)malloc(1);
    p[0] = 0;
    free((void*)p);
    sched_yield();
  }
}

int main() {
  barrier_init(&barrier, 2);
  mainth = pthread_self();
  struct sigaction act = {};
  act.sa_sigaction = &MyHandler;
  sigaction(SIGPROF, &act, 0);
  pthread_t th;
  pthread_create(&th, 0, sendsignal, 0);
  loop();
  pthread_join(th, 0);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: signal handler spoils errno
// CHECK:   Signal 27 handler invoked at:
// CHECK:     #0 MyHandler(int, {{(__)?}}siginfo{{(_t)?}}*, void*) {{.*}}signal_errno.cpp
// CHECK:     main
// CHECK: SUMMARY: ThreadSanitizer: signal handler spoils errno{{.*}}MyHandler
