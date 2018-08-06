// RUN: %clang_cc1 -std=c89 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=gnu89 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c99 -fsyntax-only -verify %s -DC99

#ifdef C99
// expected-no-diagnostics
#endif

void foo() {
#ifndef C99
  // expected-warning@+2{{GCC does not allow variable declarations in for loop initializers before C99}}
#endif
  for (int i = 0; i < 10; i++)
    ;
}
