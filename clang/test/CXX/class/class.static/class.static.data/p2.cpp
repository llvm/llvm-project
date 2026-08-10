// RUN: %clang_cc1 -std=c++1z -verify %s

struct X {
  static struct A a;
  static inline struct B b; // expected-error {{incomplete type}} expected-note {{forward decl}}
  static inline struct C c = {}; // expected-error {{incomplete type}} expected-note {{forward decl}}
};
