// Check that -w has higher priority than -Werror.
// RUN: %clang_cc1 -verify -Wsign-compare -Werror -w %s
// expected-no-diagnostics

int f0(int x, unsigned y) {
  return x < y;
}
