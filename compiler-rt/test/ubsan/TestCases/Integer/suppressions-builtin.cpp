// RUN: %clangxx -fsanitize=builtin -g0 %s -o %t

// Suppression by symbol name requires the compiler-rt runtime to be able to
// symbolize stack addresses.
// REQUIRES: can-symbolize
// UNSUPPORTED: android

// RUN: echo "invalid-builtin-use:do_ctz" > %t.func-supp
// RUN: %env_ubsan_opts=halt_on_error=1:suppressions='"%t.func-supp"' %run %t

#include <stdint.h>

extern "C" void do_ctz(int n) { __builtin_ctz(0); }

int main() {
  do_ctz(0);
  return 0;
}
