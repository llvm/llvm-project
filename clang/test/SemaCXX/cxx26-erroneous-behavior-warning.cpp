// RUN: %clang_cc1 -std=c++26 -fsyntax-only -Wuninitialized -verify %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -Wuninitialized -verify=cxx23 %s

// Test for C++26 erroneous behavior diagnostics (P2795R5)
// In C++26, reading an uninitialized local variable without [[indeterminate]]
// is erroneous behavior and produces an error.
// In C++23 and earlier, it's the usual -Wuninitialized warning.

void test_uninit_read() {
  int x;              // expected-note {{initialize the variable 'x' to silence this warning}} \
                      // cxx23-note {{initialize the variable 'x' to silence this warning}}
  int y = x;          // expected-error {{variable 'x' is uninitialized when used here}} \
                      // cxx23-warning {{variable 'x' is uninitialized when used here}}
}

#if __cplusplus >= 202400L
// With [[indeterminate]], the user explicitly opts into indeterminate values,
// so the diagnostic is suppressed entirely.
void test_indeterminate_no_warning() {
  [[indeterminate]] int x;
  int y = x;          // no diagnostic
}
#endif
