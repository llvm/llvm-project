// RUN: %clang_cc1 -std=c++20 -verify %s

namespace GH57347 {
namespace A {}

void f() {
  using enum A::+; // expected-error {{using enum requires an enum or typedef name}}
  using enum; // expected-error {{using enum requires an enum or typedef name}}
  using enum class; // expected-error {{using enum requires an enum or typedef name}}
  using enum enum q; // expected-error {{using enum does not permit an elaborated enum specifier}}
}
}
