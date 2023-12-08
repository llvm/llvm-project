// RUN: %clang_cc1 -fsyntax-only -Wunused -verify %s

// Make sure 'unused' warnings are disabled when errors occurred.
static void foo(int *X) { // expected-note {{candidate}}
}
void bar(const int *Y) {
  foo(Y); // expected-error {{no matching function for call}}
}
