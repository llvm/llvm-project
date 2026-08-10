// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"
#include <signal.h>
#include <sys/types.h>

static void handler(int, siginfo_t *, void *) {
  write(2, "SIGNAL\n", 7);
  // CHECK: SIGNAL
  _exit(0);
  // CHECK-NOT: ThreadSanitizer: signal-unsafe call
}

int main() {
  struct sigaction act = {};
  act.sa_sigaction = &handler;
  act.sa_flags = SA_SIGINFO;
  sigaction(SIGPROF, &act, 0);
  raise(SIGPROF);
  fprintf(stderr, "DONE\n");
  // CHECK-NOT: DONE
  return 0;
}
