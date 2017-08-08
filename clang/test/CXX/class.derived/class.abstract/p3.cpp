// RUN: %clang_cc1 -std=c++1z -verify %s

struct A {
  A() {}
  A(int) : A() {} // ok

  virtual void f() = 0; // expected-note 1+{{unimplemented}}
};

template<typename> struct SecretlyAbstract {
  SecretlyAbstract();
  SecretlyAbstract(int);
  virtual void f() = 0; // expected-note 1+{{unimplemented}}
};
using B = SecretlyAbstract<int>;
using C = SecretlyAbstract<float>;
using D = SecretlyAbstract<char>[1];

B b; // expected-error {{abstract class}}
D d; // expected-error {{abstract class}}

template<int> struct N;

// Note: C is not instantiated anywhere in this file, so we never discover that
// it is in fact abstract. The C++ standard suggests that we need to
// instantiate in all cases where abstractness could affect the validity of a
// program, but that breaks a *lot* of code, so we don't do that.
//
// FIXME: Once DR1640 is resolved, remove the check on forming an abstract
// array type entirely. The only restriction we need is that you can't create
// an object of abstract (most-derived) type.


// An abstract class shall not be used

//  - as a parameter type
void f(A&);
void f(A); // expected-error {{abstract class}}
void f(A[1]); // expected-error {{abstract class}}
void f(B); // expected-error {{abstract class}}
void f(B[1]); // expected-error {{abstract class}}
void f(C);
void f(C[1]);
void f(D); // expected-error {{abstract class}}
void f(D[1]); // expected-error {{abstract class}}

//  - as a function return type
A &f(N<0>);
A *f(N<1>);
A f(N<2>); // expected-error {{abstract class}}
A (&f(N<3>))[2]; // expected-error {{abstract class}}
B f(N<4>); // expected-error {{abstract class}}
B (&f(N<5>))[2]; // expected-error {{abstract class}}
C f(N<6>);
C (&f(N<7>))[2];

//  - as the type of an explicit conversion
void g(A&&);
void h() {
  A(); // expected-error {{abstract class}}
  A(0); // expected-error {{abstract class}}
  A{}; // expected-error {{abstract class}}
  A{0}; // expected-error {{abstract class}}
  (A)(0); // expected-error {{abstract class}}
  (A){}; // expected-error {{abstract class}}
  (A){0}; // expected-error {{abstract class}}

  D(); // expected-error {{array type}}
  D{}; // expected-error {{abstract class}}
  D{0}; // expected-error {{abstract class}}
  (D){}; // expected-error {{abstract class}}
  (D){0}; // expected-error {{abstract class}}
}

template<typename T> void t(T); // expected-note 2{{abstract class}}
void i(A &a, B &b, C &c, D &d) {
  // FIXME: These should be handled consistently. We currently reject the first
  // two early because we (probably incorrectly, depending on dr1640) take
  // abstractness into account in forming implicit conversion sequences.
  t(a); // expected-error {{no matching function}}
  t(b); // expected-error {{no matching function}}
  t(c); // expected-error {{allocating an object of abstract class type}}
  t(d); // ok, decays to pointer
}

struct E : A {
  E() : A() {} // ok
  E(int n) : A( A(n) ) {} // expected-error {{abstract class}}
};

namespace std {
  template<typename T> struct initializer_list {
    const T *begin, *end;
    initializer_list();
  };
}
std::initializer_list<A> ila = {1, 2, 3, 4}; // expected-error {{abstract class}}
