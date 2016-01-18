// RUN: %clang_cc1 -fsyntax-only -verify %s

static void test() {
  int *pi;
  int x;
  typeof pi[x] y; 
}

// Part of rdar://problem/8347416;  from the gcc test suite.
struct S {
  int i;
  __typeof(S::i) foo(); // expected-error {{invalid use of non-static data member 'i'}}
};
