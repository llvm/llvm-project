// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct B0;

class A {
  friend class B {}; // expected-error {{cannot define a type in a friend declaration}}
  friend int;
  friend B0;
#if __cplusplus <= 199711L
  // expected-warning@-2 {{unelaborated friend declaration is a C++11 extension; specify 'struct' to befriend 'B0'}}
#endif
  friend class C; // okay
};
