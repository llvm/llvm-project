// RUN: %clang_cc1 -fsyntax-only -verify %s
// C++0x N2914.

struct X {
  int i;
  static int a;
};

using X::i; // expected-error{{using declaration can not refer to class member}}
using X::s; // expected-error{{using declaration can not refer to class member}}

void f() {
  using X::i; // expected-error{{using declaration can not refer to class member}}
  using X::s; // expected-error{{using declaration can not refer to class member}}
}
