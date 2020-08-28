// RUN: %clangxx_msan -fno-sanitize=memory -c %s -o %t-main.o
// RUN: %clangxx_msan %t-main.o %s -o %t
// RUN: %run %t

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <ucontext.h>
#include <unistd.h>

#include <sanitizer/msan_interface.h>

#if __has_feature(memory_sanitizer)

__attribute__((noinline)) int bar(int a, int b) {
  volatile int zero = 0;
  return zero;
}

void foo(int x, int y, int expected) {
  assert(__msan_test_shadow(&x, sizeof(x)) == expected);
  assert(__msan_test_shadow(&y, sizeof(y)) == expected);

  // Poisons parameter shadow in TLS so that the next call (to foo) from
  // uninstrumented main has params 1 and 2 poisoned no matter what.
  int a, b;
  (void)bar(a, b);
}

#else

// This code is not instrumented by MemorySanitizer to prevent it from modifying
// MSAN TLS data for this test.

int foo(int, int, int);

int main(int argc, char **argv) {
  int x, y;
  // The parameters should _not_ be poisoned; this is the first call to foo.
  foo(x, y, -1);
  // The parameters should be poisoned; the prior call to foo left them so.
  foo(x, y, 0);

  ucontext_t ctx;
  if (getcontext(&ctx) == -1) {
    perror("getcontext");
    _exit(1);
  }

  // Simulate a fiber switch occurring from MSAN's perspective (though no switch
  // actually occurs).
  const void *previous_stack_bottom = nullptr;
  size_t previous_stack_size = 0;
  __msan_start_switch_fiber(ctx.uc_stack.ss_sp, ctx.uc_stack.ss_size);
  __msan_finish_switch_fiber(&previous_stack_bottom, &previous_stack_size);

  // The simulated fiber switch will reset the TLS parameter shadow. So even
  // though the most recent call to foo left the parameter shadow poisoned, the
  // parameters are _not_ expected to be poisoned now.
  foo(x, y, -1);

  return 0;
}

#endif
