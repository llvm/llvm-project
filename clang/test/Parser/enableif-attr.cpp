// RUN: %clang_cc1 -x c -fsyntax-only -verify %s

void f1(x) __attribute__((enable_if(1, "")));    // expected-error {{a parameter list without types is only allowed in a function definition}}
void f2(x, y) __attribute__((enable_if(1, ""))); // expected-error {{a parameter list without types is only allowed in a function definition}}
