// RUN: %clang_cc1 -std=c++26 -fsyntax-only -Wuninitialized -verify %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -Wuninitialized -verify %s

int uninitialized_read() {
  int x;    // expected-note {{initialize the variable 'x' to silence this warning}}
  return x; // expected-warning {{variable 'x' is uninitialized when used here}}
}

#if __cplusplus > 202302L
void indeterminate_read() {
  [[indeterminate]] int x;
  int y = x;
}
#endif
