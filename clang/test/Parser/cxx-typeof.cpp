// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=gnu++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=gnu++11 %s

static void test() {
  int *pi;
  int x;
  typeof pi[x] y; 
}

// From the gcc test suite.
struct S {
  int i;
  __typeof(S::i) foo();
#if __cplusplus <= 199711L
  // expected-error@-2 {{invalid use of non-static data member 'i'}}
#else
  // expected-no-diagnostics
#endif
};
