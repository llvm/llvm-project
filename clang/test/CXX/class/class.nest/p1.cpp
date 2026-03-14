// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

class Outer {
  int x;
  static int sx;
  int f();

  // C++11 does relax this rule (see 5.1.1.10) in the first case, but we need to enforce it in C++03 mode.
  class Inner {
    static char a[sizeof(x)];
#if __cplusplus <= 199711L
    // expected-error@-2 {{invalid use of non-static data member 'x'}}
#endif
    static char b[sizeof(sx)]; // okay
    static char c[sizeof(f)]; // expected-error {{call to non-static member function without an object argument}}
  };
};
