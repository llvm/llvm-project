// Check that #pragma diagnostic warning overrides -Werror.
//
// RUN: %clang_cc1 -verify -Werror %s

#pragma clang diagnostic warning "-Wsign-compare"
int f0(int x, unsigned y) {
  return x < y; // expected-warning {{comparison of integers}}
}
