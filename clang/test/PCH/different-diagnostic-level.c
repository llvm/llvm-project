// RUN: %clang -x c-header %s -Weverything -o %t.h.pch
// RUN: %clang -x c %s -w -include %t.h -fsyntax-only -Xclang -verify

#ifndef HEADER
#define HEADER

extern int foo;

#else

void f(void) {
  int a = foo;
  // Make sure we parsed this by getting an error.
  int b = bar; // expected-error {{undeclared}}
}

#endif
