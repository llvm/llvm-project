// Check that "#pragma diagnostic error" is suppressed by -w.
//
// RUN: %clang_cc1 -verify -Werror -w %s

// expected-no-diagnostics
#pragma gcc diagnostic error "-Wsign-compare"
int f0(int x, unsigned y) {
  return x < y;
}
