// RUN: %clang_cc1 -fsyntax-only -Wno-tautological-pointer-compare -fblocks -std=c++11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-tautological-pointer-compare -fblocks -std=c++11 -verify %s -fexperimental-new-constant-interpreter

void foo() {
  int a;
  bool b;

  a = 0 ? nullptr + a : a + nullptr; // expected-error 2{{invalid operands to binary expression}}
  a = 0 ? nullptr - a : a - nullptr; // expected-error 2{{invalid operands to binary expression}}
  a = 0 ? nullptr / a : a / nullptr; // expected-error 2{{invalid operands to binary expression}}
  a = 0 ? nullptr * a : a * nullptr; // expected-error 2{{invalid operands to binary expression}}
  a = 0 ? nullptr >> a : a >> nullptr; // expected-error 2{{invalid operands to binary expression}}
  a = 0 ? nullptr << a : a << nullptr; // expected-error 2{{invalid operands to binary expression}}
  a = 0 ? nullptr % a : a % nullptr; // expected-error 2{{invalid operands to binary expression}}
  a = 0 ? nullptr & a : a & nullptr; // expected-error 2{{invalid operands to binary expression}}
  a = 0 ? nullptr | a : a | nullptr; // expected-error 2{{invalid operands to binary expression}}
  a = 0 ? nullptr ^ a : a ^ nullptr; // expected-error 2{{invalid operands to binary expression}}

  // Using two nullptrs should only give one error instead of two.
  a = nullptr + nullptr; // expected-error{{invalid operands to binary expression}}
  a = nullptr - nullptr; // expected-error{{invalid operands to binary expression}}
  a = nullptr / nullptr; // expected-error{{invalid operands to binary expression}}
  a = nullptr * nullptr; // expected-error{{invalid operands to binary expression}}
  a = nullptr >> nullptr; // expected-error{{invalid operands to binary expression}}
  a = nullptr << nullptr; // expected-error{{invalid operands to binary expression}}
  a = nullptr % nullptr; // expected-error{{invalid operands to binary expression}}
  a = nullptr & nullptr; // expected-error{{invalid operands to binary expression}}
  a = nullptr | nullptr; // expected-error{{invalid operands to binary expression}}
  a = nullptr ^ nullptr; // expected-error{{invalid operands to binary expression}}

  a += nullptr; // expected-error{{invalid operands to binary expression}}
  a -= nullptr; // expected-error{{invalid operands to binary expression}}
  a /= nullptr; // expected-error{{invalid operands to binary expression}}
  a *= nullptr; // expected-error{{invalid operands to binary expression}}
  a >>= nullptr; // expected-error{{invalid operands to binary expression}}
  a <<= nullptr; // expected-error{{invalid operands to binary expression}}
  a %= nullptr; // expected-error{{invalid operands to binary expression}}
  a &= nullptr; // expected-error{{invalid operands to binary expression}}
  a |= nullptr; // expected-error{{invalid operands to binary expression}}
  a ^= nullptr; // expected-error{{invalid operands to binary expression}}

  b = a < nullptr || nullptr < a; // expected-error 2{{invalid operands to binary expression}}
  b = a > nullptr || nullptr > a; // expected-error 2{{invalid operands to binary expression}}
  b = a <= nullptr || nullptr <= a; // expected-error 2{{invalid operands to binary expression}}
  b = a >= nullptr || nullptr >= a; // expected-error 2{{invalid operands to binary expression}}
  b = a == nullptr || nullptr == a; // expected-error 2{{invalid operands to binary expression}}
  b = a != nullptr || nullptr != a; // expected-error 2{{invalid operands to binary expression}}

  b = &a < nullptr || nullptr < &a || &a > nullptr || nullptr > &a; // expected-error 4{{invalid operands}}
  b = &a <= nullptr || nullptr <= &a || &a >= nullptr || nullptr >= &a; // expected-error 4{{invalid operands}}
  b = &a == nullptr || nullptr == &a || &a != nullptr || nullptr != &a;

  b = nullptr < nullptr || nullptr > nullptr; // expected-error 2{{invalid operands to binary expression}}
  b = nullptr <= nullptr || nullptr >= nullptr; // expected-error 2{{invalid operands to binary expression}}
  b = nullptr == nullptr || nullptr != nullptr;

  b = ((nullptr)) != a;  // expected-error{{invalid operands to binary expression}}

  void (^c)();
  c = nullptr;
  b = c == nullptr || nullptr == c || c != nullptr || nullptr != c;
  
  class X;
  void (X::*d) ();
  d = nullptr;
  b = d == nullptr || nullptr == d || d != nullptr || nullptr != d;

  extern void e();
  b = e == nullptr || nullptr == e || e != nullptr || nullptr != e;

  int f[2];
  b = f == nullptr || nullptr == f || f != nullptr || nullptr != f;
  b = "f" == nullptr || nullptr == "f" || "f" != nullptr || nullptr != "f";
}
