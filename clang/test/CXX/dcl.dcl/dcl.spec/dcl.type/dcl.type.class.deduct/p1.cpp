// RUN: %clang_cc1 -std=c++1z -verify %s

template<typename T> struct A { constexpr A(int = 0) {} };
A() -> A<int>;
A(int) -> A<char>;

static constexpr inline const volatile A a = {}; // ok, specifiers are permitted
// FIXME: There isn't really a good reason to reject this.
A b; // expected-error {{requires an initializer}}
A c [[]] {};

A d = {}, e = {};
A f(0), g{}; // expected-error {{template arguments deduced as 'A<char>' in declaration of 'f' and deduced as 'A<int>' in declaration of 'g'}}

struct B {
  static A a; // expected-error {{requires an initializer}}
};
extern A x; // expected-error {{requires an initializer}}
