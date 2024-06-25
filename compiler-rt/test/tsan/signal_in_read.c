// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

#include "test.h"

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static int SignalPipeFd[] = {-1, -1};
static int BlockingPipeFd[] = {-1, -1};

static void Handler(int _) { assert(write(SignalPipeFd[1], ".", 1) == 1); }

static void *ThreadFunc(void *_) {
  char C;
  assert(read(BlockingPipeFd[0], &C, sizeof(C)) == 1);
  assert(C == '.');
  return 0;
}

int main() {
  alarm(60); // Kill the test if it hangs.

  assert(pipe(SignalPipeFd) == 0);
  assert(pipe(BlockingPipeFd) == 0);

  struct sigaction act;
  sigemptyset(&act.sa_mask);
  act.sa_flags = SA_RESTART;
  act.sa_handler = Handler;
  assert(sigaction(SIGUSR1, &act, 0) == 0);

  pthread_t Thr;
  assert(pthread_create(&Thr, 0, ThreadFunc, 0) == 0);

  // Give the thread enough time to block in the read call.
  usleep(1000000);

  // Signal the thread, this should run the signal handler and unblock the read
  // below.
  pthread_kill(Thr, SIGUSR1);
  char C;
  assert(read(SignalPipeFd[0], &C, 1) == 1);

  // Unblock the thread and join it.
  assert(write(BlockingPipeFd[1], &C, 1) == 1);
  void *_ = 0;
  assert(pthread_join(Thr, &_) == 0);

  fprintf(stderr, "PASS\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: PASS
