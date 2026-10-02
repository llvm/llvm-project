// RUN: %clang_tsan -O1 %s -o %t && %env_tsan_opts=halt_on_error=1:abort_on_error=1:handle_abort=0 %deflake %run %t 2>&1 | FileCheck %s
#include "test.h"
#include <signal.h>

int Global;

__attribute__((noinline)) static void step(int *p, int i) { *p = i; }

static void handler(int, siginfo_t *, void *) {
  // Overflow the active TracePart in the SIGABRT handler to trigger
  // TracePartAlloc, which acquires ctx->slot_mtx.
  int x = 0;
  for (int i = 0; i < 10000; ++i)
    step(&x, i);
  write(2, "SIGNAL\n", 7);
  _exit(0);
}

void *Thread(void *x) {
  Global = 42;
  barrier_wait(&barrier);
  return NULL;
}

int main() {
  struct sigaction act = {};
  act.sa_sigaction = &handler;
  act.sa_flags = SA_SIGINFO;
  sigaction(SIGABRT, &act, 0);

  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, NULL, Thread, NULL);
  pthread_detach(t);
  barrier_wait(&barrier);
  Global = 43;
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: SIGNAL
