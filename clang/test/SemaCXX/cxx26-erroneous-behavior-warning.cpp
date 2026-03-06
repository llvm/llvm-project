// RUN: %clang_cc1 -std=c++26 -fsyntax-only -Werroneous-behavior -verify %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -Wuninitialized -verify=cxx23 %s

// Test for C++26 erroneous behavior warnings (P2795R5)

void test_erroneous_read() {
  int x;              // expected-note {{variable 'x' was default-initialized here}} \
                      // expected-note {{initialize the variable 'x' to silence this warning}} \
                      // cxx23-note {{initialize the variable 'x' to silence this warning}}
  int y = x;          // expected-warning {{reading from variable 'x' with erroneous value is erroneous behavior}} \
                      // cxx23-warning {{variable 'x' is uninitialized when used here}}
}

// In C++23, this is regular uninitialized warning
void test_cxx23_uninit() {
  int x;              // cxx23-note {{initialize the variable 'x' to silence this warning}} \
                      // expected-note {{variable 'x' was default-initialized here}} \
                      // expected-note {{initialize the variable 'x' to silence this warning}}
  int y = x;          // cxx23-warning {{variable 'x' is uninitialized when used here}} \
                      // expected-warning {{reading from variable 'x' with erroneous value is erroneous behavior}}
}

#if __cplusplus >= 202400L
// With [[indeterminate]], it's still a regular uninitialized warning (UB, not erroneous)
void test_indeterminate_uninit() {
  [[indeterminate]] int x;
  int y = x;          // No erroneous behavior warning - this is UB, not erroneous
}
#endif
