/* Checks that BOLT correctly handles instrumentation of indirect calls
 * including case with indirect calls in signals handlers.
 */
#include <signal.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int foo(int x) { return x + 1; }

int bar(int (*fn)(int), int val) { return fn(val); }

void sigHandler(int signum) { bar(foo, 3); }

int main(int argc, char **argv) {
  long long i;
  pid_t pid, wpid;
  int wstatus;
  signal(SIGUSR1, sigHandler);
  pid = fork();
  if (pid) {
    do {
      kill(pid, SIGUSR1);
      usleep(0);
      wpid = waitpid(pid, &wstatus, WNOHANG);
    } while (wpid == 0);
    printf("[parent]\n");
  } else {
    for (i = 0; i < 100000; i++) {
      bar(foo, i % 10);
    }
    printf("[child]\n");
  }
  return 0;
}

/*
REQUIRES: system-linux && lit-max-individual-test-time

RUN: %clang %cflags %s -o %t.exe -Wl,-q -pie -fpie

RUN: llvm-bolt %t.exe -instrument -instrumentation-file=%t.fdata \
RUN:   -instrumentation-wait-forks=1 -conservative-instrumentation \
RUN:   -o %t.instrumented_conservative

# Instrumented program needs to finish returning zero
RUN: %t.instrumented_conservative | FileCheck %s -check-prefix=CHECK-OUTPUT

RUN: llvm-bolt %t.exe -instrument -instrumentation-file=%t.fdata \
RUN:   -instrumentation-wait-forks=1 \
RUN:   -o %t.instrumented

# Instrumented program needs to finish returning zero
RUN: %t.instrumented | FileCheck %s -check-prefix=CHECK-OUTPUT

# Test that the instrumented data makes sense
RUN:  llvm-bolt %t.exe -o %t.bolted -data %t.fdata \
RUN:    -reorder-blocks=cache+ -reorder-functions=hfsort+ \
RUN:    -print-only=interp -print-finalized

RUN: %t.bolted | FileCheck %s -check-prefix=CHECK-OUTPUT

CHECK-OUTPUT: [child]
CHECK-OUTPUT: [parent]
*/
