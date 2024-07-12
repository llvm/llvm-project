// RUN: %clang_cc1 -verify %s

struct A {
  int x;
};

void operator&(A, A);

template<typename T>
struct B {
  int f() {
    return T::x & 1; // expected-error {{invalid use of non-static data member 'x'}}
  }
};

template struct B<A>; // expected-note {{in instantiation of}}
