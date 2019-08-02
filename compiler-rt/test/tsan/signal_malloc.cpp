// RUN: %clang_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"
#include <signal.h>
#include <sys/types.h>

static void handler(int, siginfo_t*, void*) {
  // CHECK: WARNING: ThreadSanitizer: signal-unsafe call inside of a signal
  // CHECK:     #0 malloc
  // CHECK:     #{{(1|2)}} handler(int, {{(__)?}}siginfo{{(_t)?}}*, void*) {{.*}}signal_malloc.cpp:[[@LINE+2]]
  // CHECK: SUMMARY: ThreadSanitizer: signal-unsafe call inside of a signal{{.*}}handler
  volatile char *p = (char*)malloc(1);
  p[0] = 0;
  free((void*)p);
}

int main() {
  struct sigaction act = {};
  act.sa_sigaction = &handler;
  sigaction(SIGPROF, &act, 0);
  kill(getpid(), SIGPROF);
  sleep(1);  // let the signal handler run
  return 0;
}

