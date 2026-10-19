// RUN: %clang_tsan -O1 %s -o %t && %env_tsan_opts=halt_on_error=1:abort_on_error=1:handle_abort=0 not %run %t 2>&1 | FileCheck %s
#include "test.h"
#include <signal.h>

int Global;

__attribute__((noinline)) void step() { asm volatile(""); }

static void handler(int, siginfo_t *, void *) {
  // Overflow the active 256KB TracePart (~32K events) in the SIGABRT handler
  // to trigger TraceSwitchPartImpl -> TracePartAlloc, which acquires
  // ctx->slot_mtx.
  for (int i = 0; i < 100000; ++i)
    step();
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
