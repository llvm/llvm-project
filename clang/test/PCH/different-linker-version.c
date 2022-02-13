// RUN: %clang_cc1 -target-linker-version 100 -emit-pch %s -o %t.h.pch
// RUN: %clang_cc1 -target-linker-version 200 %s -include-pch %t.h.pch -fsyntax-only -verify

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
