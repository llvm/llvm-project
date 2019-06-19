// RUN: %clang_hwasan -O1 %s -o %t
// RUN: %env_hwasan_opts=stack_history_size=2048 not %run %t 2045 2>&1 | FileCheck %s --check-prefix=YES
// RUN: %env_hwasan_opts=stack_history_size=2048 not %run %t 2047 2>&1 | FileCheck %s --check-prefix=NO

// REQUIRES: stable-runtime

#include <stdlib.h>

void USE(void *x) { // pretend_to_do_something(void *x)
  __asm__ __volatile__("" : : "r" (x) : "memory");
}

volatile int four = 4;
__attribute__((noinline)) void FUNC0() { int x[4]; USE(&x[0]); }
__attribute__((noinline)) void FUNC() { int x[4]; USE(&x[0]); }
__attribute__((noinline)) void OOB() { int x[4]; x[four] = 0; USE(&x[0]); }

int main(int argc, char **argv) {
  int X = argc == 2 ? atoi(argv[1]) : 10;
  // FUNC0 is X+2's element of the ring buffer.
  // If runtime buffer size is less than it, FUNC0 record will be lost.
  FUNC0();
  for (int i = 0; i < X; ++i)
    FUNC();
  // Make at least one call to OOB where base tag != 0 so that the bug is caught
  // at least once.
  OOB();
  OOB();
}

// YES: Previously allocated frames
// YES: OOB
// YES: FUNC
// YES: FUNC0

// NO: Previously allocated frames
// NO: OOB
// NO: FUNC
// NO-NOT: FUNC0
