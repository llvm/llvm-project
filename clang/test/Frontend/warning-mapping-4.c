// Verify that various combinations of flags properly keep the sign-compare
// warning disabled.

// RUN: %clang_cc1 -verify -Wno-error=sign-compare %s
// RUN: %clang_cc1 -verify -Wsign-compare -w -Wno-error=sign-compare %s
// RUN: %clang_cc1 -verify -w -Werror=sign-compare %s
// expected-no-diagnostics

int f0(int x, unsigned y) {
  return x < y;
}
