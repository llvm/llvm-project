// RUN: %clang_cc1 -fsyntax-only -verify %s

struct B0;

class A {
  friend class B {}; // expected-error {{cannot define a type in a friend declaration}}
  friend int; // expected-warning {{non-class friend type 'int' is a C++11 extension}}
  friend B0; // expected-warning {{specify 'struct' to befriend 'B0'}}
  friend class C; // okay
};
