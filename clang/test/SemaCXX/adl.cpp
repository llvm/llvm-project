// RUN: %clang_cc1 -std=c++11 -verify %s

namespace PR40329 {
  struct A {
    A(int);
    friend int operator->*(A, A);
  };
  struct B : A {
    B();
    enum E { e };
  };
  // Associated classes for B are {B, A}
  // Associated classes for B::E are {B} (non-transitive in this case)
  //
  // If we search B::E first, we must not mark B "visited" and shortcircuit
  // visiting it later, or we won't find the associated class A.
  int k0 = B::e ->* B::e; // expected-error {{non-pointer-to-member type}}
  int k1 = B::e ->* B();
  int k2 = B() ->* B::e;
}
