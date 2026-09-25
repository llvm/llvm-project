// REQUIRES: ubsan-standalone
// REQUIRES: target={{(i.86|x86_64)-.*-linux.*}}
// RUN: %clangxx -O2 -fsanitize=signed-integer-overflow -fsanitize-trap=signed-integer-overflow -fsanitize-trap-loop %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s
#include <assert.h>
#include <limits.h>
#include <sanitizer/ubsan_interface.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

__attribute__((noinline)) static void poison_stack() {
  volatile unsigned char buf[256];
  for (size_t i = 0; i < sizeof(buf); ++i)
    buf[i] = 0xaa;
}

static void sigill_handler(int) {
  const char msg[] = "CAUGHT_TRAP_LOOP_SIGILL\n";
  write(STDERR_FILENO, msg, sizeof(msg) - 1);
  _exit(0);
}

__attribute__((noinline)) int trigger_overflow(int a, int b) { return a + b; }

int main() {
  struct sigaction sa_ill = {};
  sa_ill.sa_handler = sigill_handler;
  sigaction(SIGILL, &sa_ill, nullptr);
  poison_stack();
  __ubsan_install_trap_loop_detection();

  struct sigaction old_sa = {};
  assert(sigaction(SIGPROF, nullptr, &old_sa) == 0);
  assert((old_sa.sa_flags & SA_SIGINFO) != 0);
  assert((old_sa.sa_flags & SA_RESETHAND) == 0);
  assert(!sigismember(&old_sa.sa_mask, SIGILL));

  // Verify multiple SIGPROF deliveries during normal execution do not crash
  // (e.g. on i386 when SA_SIGINFO is missing) or reset SIGPROF to SIG_DFL
  // (when SA_RESETHAND is set in uninitialized sa_flags).
  raise(SIGPROF);
  raise(SIGPROF);
  fprintf(stderr, "SURVIVED_NORMAL_SIGPROF\n");

  // Trigger a trap loop and verify SIGPROF -> __builtin_trap() -> SIGILL
  // reaches sigill_handler without SIGILL being blocked by uninitialized sa_mask.
  // CHECK: SURVIVED_NORMAL_SIGPROF
  // CHECK: CAUGHT_TRAP_LOOP_SIGILL
  return trigger_overflow(INT_MAX, 1);
}
