// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

class Outer {
  int x;
  static int sx;
  int f();

  // The first case is invalid in the C++03 mode but valid in C++0x (see 5.1.1.10).
  class Inner {
    static char a[sizeof(x)]; // okay
    static char b[sizeof(sx)]; // okay
    static char c[sizeof(f)]; // expected-error {{call to non-static member function without an object argument}}
  };
};
