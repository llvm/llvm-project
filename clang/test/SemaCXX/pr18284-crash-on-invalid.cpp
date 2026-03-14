// RUN: %clang_cc1 -fsyntax-only -verify %s
// Don't crash (PR18284).

namespace n1 {
class A { };
class C { A a; };

A::RunTest() {} // expected-error {{a type specifier is required for all declarations}}

void f() {
  new C;
}
} // namespace n1

namespace n2 {
class A { };
class C : public A { };

A::RunTest() {} // expected-error {{a type specifier is required for all declarations}}

void f() {
  new C;
}
} // namespace n2
