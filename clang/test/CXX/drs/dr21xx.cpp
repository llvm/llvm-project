// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr2180 { // dr2180: yes
  class A {
    A &operator=(const A &); // expected-note 0-2{{here}}
    A &operator=(A &&); // expected-note 0-2{{here}} expected-error 0-1{{extension}}
  };

  struct B : virtual A {
    B &operator=(const B &);
    B &operator=(B &&); // expected-error 0-1{{extension}}
    virtual void foo() = 0;
  };
#if __cplusplus < 201103L
  B &B::operator=(const B&) = default; // expected-error {{private member}} expected-error {{extension}} expected-note {{here}}
  B &B::operator=(B&&) = default; // expected-error {{private member}} expected-error 2{{extension}} expected-note {{here}}
#else
  B &B::operator=(const B&) = default; // expected-error {{would delete}} expected-note@-9{{inaccessible copy assignment}}
  B &B::operator=(B&&) = default; // expected-error {{would delete}} expected-note@-10{{inaccessible move assignment}}
#endif
}
