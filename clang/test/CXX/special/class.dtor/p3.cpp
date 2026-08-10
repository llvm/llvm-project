// RUN: %clang_cc1 -std=c++11 -fcxx-exceptions -verify %s

// The exception specification of a destructor declaration is matched *before*
// the exception specification adjustment occurs.
namespace DR1492 {
  struct A { ~A(); }; // expected-note {{here}}
  A::~A() noexcept {} // expected-warning {{previously declared with an implicit exception specification}}

  struct B { ~B() noexcept; }; // expected-note {{here}}
  B::~B() {} // expected-warning {{previously declared with an explicit exception specification}}

  template<typename T> struct C {
    T t;
    ~C(); // expected-note {{here}}
  };
  template<typename T> C<T>::~C() noexcept {} // expected-error {{does not match previous}}
}
