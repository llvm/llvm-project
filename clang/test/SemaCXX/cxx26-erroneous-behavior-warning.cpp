// RUN: %clang_cc1 -std=c++26 -fsyntax-only -Wuninitialized -verify %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -Wuninitialized -verify %s

// P2795R5 erroneous-behavior diagnostics are not yet implemented; reading an
// uninitialized local variable is still the usual -Wuninitialized warning in
// both C++23 and C++26.

int test_uninit_read() {
  int x;              // expected-note {{initialize the variable 'x' to silence this warning}}
  return x;           // expected-warning {{variable 'x' is uninitialized when used here}}
}

#if __cplusplus >= 202400L
// With [[indeterminate]], the user explicitly opts into indeterminate values,
// so the diagnostic is suppressed entirely.
void test_indeterminate_no_warning() {
  [[indeterminate]] int x;
  int y = x;          // no diagnostic
}
#endif
