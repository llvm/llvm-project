// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -Wunused-value %s

void f() __attribute__((const));

namespace PR18571 {
// Unevaluated contexts should not trigger unused result warnings.
template <typename T>
auto foo(T) -> decltype(f(), bool()) { // Should not warn.
  return true;
}

void g() {
  foo(1);
}

void h() {
  int i = 0;
  (void)noexcept(++i); // expected-warning {{expression with side effects has no effect in an unevaluated context}}
  decltype(i++) j = 0; // expected-warning {{expression with side effects has no effect in an unevaluated context}}
}

struct S {
  S operator++(int);
  S(int i);
  S();

  int& f();
  S g();
};

void j() {
  S s;
  int i = 0;
  (void)noexcept(s++); // Ok
  (void)noexcept(i++); // expected-warning {{expression with side effects has no effect in an unevaluated context}}
  (void)noexcept(i = 5); // expected-warning {{expression with side effects has no effect in an unevaluated context}}
  (void)noexcept(s = 5); // Ok

  (void)sizeof(s.f()); // Ok
  (void)sizeof(s.f() = 5); // expected-warning {{expression with side effects has no effect in an unevaluated context}}
  (void)noexcept(s.g() = 5); // Ok
}

}