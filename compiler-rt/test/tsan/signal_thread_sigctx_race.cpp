// RUN: %clangxx_tsan %s -o %t && %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: darwin

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/select.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

// This attempts to exercise a race condition where both a thread and its signal
// handler allocate the SigCtx. If the race is allowed, it leads to a leak and
// the first signal being dropped.
// Spawn threads in a loop and send it SIGUSR1 concurrently with the thread
// doing a bogus kill() call. The signal handler writes to a self-pipe which the
// thread detects and then exits. A dropped signal results in a timeout.
int pipes[2];
static void handler(int sig) { write(pipes[1], "x", 1); }

static int do_select() {
  struct timeval tvs {
    0, 1000
  };
  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(pipes[0], &fds);
  return select(pipes[0] + 1, &fds, 0, 0, &tvs);
}

static void *thr(void *p) {
  // This kill() is expected to fail; it exists only to trigger a call to SigCtx
  // outside of the signal handler.
  kill(INT_MIN, 0);
  int success = 0;
  for (int i = 0; i < 1024; i++) {
    if (do_select() > 0) {
      success = 1;
      break;
    }
  }
  if (success) {
    char c;
    read(pipes[0], &c, 1);
  } else {
    fprintf(stderr, "Failed to receive signal\n");
    exit(1);
  }
  return p;
}

int main() {
  if (pipe(pipes)) {
    perror("pipe");
    exit(1);
  }

  struct sigaction act = {};
  act.sa_handler = &handler;
  if (sigaction(SIGUSR1, &act, 0)) {
    perror("sigaction");
    exit(1);
  }

  for (int i = 0; i < (1 << 10); i++) {
    pthread_t th{};
    if (pthread_create(&th, 0, thr, 0)) {
      perror("pthread_create");
      exit(1);
    }
    pthread_kill(th, SIGUSR1);
    pthread_join(th, 0);
  }

  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: DONE
// CHECK-NOT: WARNING: ThreadSanitizer:
