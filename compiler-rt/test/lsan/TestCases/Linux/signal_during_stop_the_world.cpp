// Test StopTheWorld behavior during signal storm.
// Historically StopTheWorld crashed because did not handle EINTR properly.
// The test is somewhat convoluted, but that's what caused crashes previously.

// RUN: %clangxx_lsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <time.h>
#include <pthread.h>
#include <sanitizer/lsan_interface.h>

static void handler(int signo);
static void *thr(void *arg);

int main() {
  struct sigaction act = {};
  act.sa_handler = handler;
  sigaction(SIGPROF, &act, 0);

  pid_t pid = fork();
  if (pid < 0) {
    fprintf(stderr, "failed to fork\n");
    exit(1);
  }
  if (pid == 0) {
    // Child constantly sends signals to parent to cause spurious return from
    // waitpid in StopTheWorld.
    prctl(PR_SET_PDEATHSIG, SIGTERM, 0, 0, 0);
    pid_t parent = getppid();
    for (;;) {
      // There is no strong reason for these two particular signals,
      // but at least one of them ought to unblock waitpid.
      kill(parent, SIGCHLD);
      kill(parent, SIGPROF);
    }
  }
  usleep(10000);  // Let the child start.
  __lsan_do_leak_check();
  // Kill and join the child.
  kill(pid, SIGTERM);
  waitpid(pid, 0, 0);
  sleep(1);  // If the tracer thread still runs, give it time to crash.
  fprintf(stderr, "DONE\n");
// CHECK: DONE
}

static void handler(int signo) {
}

static void *thr(void *arg) {
  for (;;)
    sleep(1);
  return 0;
}
