// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

static void handler(int, siginfo_t*, void*) {
  const char *str = "HELLO FROM SIGNAL\n";
  write(2, str, strlen(str));
}

int main() {
  struct sigaction act = {};
  act.sa_sigaction = &handler;
  sigaction(SIGPROF, &act, 0);
  kill(getpid(), SIGPROF);
  sleep(1);  // let the signal handler run, can't use barrier in sig handler
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer
// CHECK: HELLO FROM SIGNAL
// CHECK: DONE

