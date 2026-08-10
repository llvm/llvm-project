// RUN: %clangxx_msan -O2 %s -o %t && %run %t

#include <sanitizer/msan_interface.h>

struct S {
  S(int a0) : a(a0) {}
  int a;
  int b;
};

// Here S is passed to FooRun as a 64-bit integer.
// This triggers an optimization where 10000 * s.a is transformed into
// ((*(uint64_t *)&s) * (10000 * 2**32)) >> 32
// Test that MSan understands that this kills the uninitialized high half of S
// (i.e. S::b).
void FooRun(S s) {
  int64_t x = 10000 * s.a;
  __msan_check_mem_is_initialized(&x, sizeof(x));
}

int main(void) {
  S z(1);
  // Take &z to ensure that it is built on stack.
  S *volatile p = &z;
  FooRun(z);
  return 0;
}
