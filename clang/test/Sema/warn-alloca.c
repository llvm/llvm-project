// RUN: %clang_cc1 -DSILENCE -fsyntax-only -verify -Wall %s
// RUN: %clang_cc1 -fsyntax-only -verify -Walloca %s

#ifdef SILENCE
  // expected-no-diagnostics
#endif

void test1(int a) {
  __builtin_alloca(a);
#ifndef SILENCE
  // expected-warning@-2 {{use of function '__builtin_alloca' is discouraged; there is no way to check for failure but failure may still occur, resulting in a possibly exploitable security vulnerability}}
#endif
}

void test2(int a) {
  __builtin_alloca_with_align(a, 32);
#ifndef SILENCE
  // expected-warning@-2 {{use of function '__builtin_alloca_with_align' is discouraged; there is no way to check for failure but failure may still occur, resulting in a possibly exploitable security vulnerability}}
#endif
}
