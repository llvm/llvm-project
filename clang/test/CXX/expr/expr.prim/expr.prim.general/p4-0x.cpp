// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct S {
  S *p = this; // ok
  decltype(this) q; // expected-error {{invalid use of 'this' outside of a non-static member function}}

  int arr[sizeof(this)]; // expected-error {{invalid use of 'this' outside of a non-static member function}}
  int sz = sizeof(this); // ok

  typedef auto f() -> decltype(this); // expected-error {{invalid use of 'this' outside of a non-static member function}}
};

namespace CaptureThis {
  struct X {
    int n = 10;
    int m = [&]{return n + 1; }();
    int o = [&]{return this->m + 1; }();
    int p = [&]{return [&](int x) { return this->m + x;}(o); }();
  };
  
  X x;
}
