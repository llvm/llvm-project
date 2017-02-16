// RUN: %clang_pgogen -O2 -o %t %s
// RUN: rm -rf default_*.profraw
// RUN: %run %t && sleep 1
// RUN: llvm-profdata show default_*.profraw 2>&1 | FileCheck %s

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <unistd.h>

#define FAKE_COUNT_SZ 2000000
/* fake counts to increse the profile size. */
unsigned long long __attribute__((section("__llvm_prf_cnts")))
counts[FAKE_COUNT_SZ];

int main(int argc, char **argv) {
  pid_t pid = fork();
  if (pid == 0) {
    int i;
    int sum = 0;
    /* child process: sleep 500us and get to runtime before the
     * main process exits. */
    prctl(PR_SET_PDEATHSIG, SIGKILL);
    usleep(500);
    for (i = 0; i < 5000; ++i)
      sum += i * i * i;
    printf("child process (%d): sum=%d\n", getpid(), sum);
  } else if (pid > 0) {
    /* parent process: sleep 100us to get into profile runtime first. */
    usleep(100);
  }
  return 0;
}

// CHECK-NOT: Empty raw profile file
