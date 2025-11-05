// RUN: %clang_cc1 -std=c11 -Wno-unused-variable -Wno-deprecated-non-prototype -Wno-inline -pedantic-errors -verify %s

inline void f(void) { // expected-note {{use 'static' to give inline function 'f' internal linkage}}
  static int x; // expected-error {{non-constant static local variable in inline function may be different in different files}}
}

