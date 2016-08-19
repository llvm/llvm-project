// Check that stores in signal handlers are not recorded in origin history.
// This is, in fact, undesired behavior caused by our chained origins
// implementation being not async-signal-safe.

// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -O3 %s -o %t && \
// RUN:     not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_msan -mllvm -msan-instrumentation-with-call-threshold=0 -fsanitize-memory-track-origins=2 -O3 %s -o %t && \
// RUN:     not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <signal.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

volatile int x, y;

void SignalHandler(int signo) {
  y = x;
}

int main(int argc, char *argv[]) {
  int volatile z;
  x = z;

  signal(SIGHUP, SignalHandler);
  kill(getpid(), SIGHUP);
  signal(SIGHUP, SIG_DFL);

  return y;
}

// CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
// CHECK-NOT: in SignalHandler
