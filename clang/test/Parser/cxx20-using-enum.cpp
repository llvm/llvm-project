// RUN: %clang_cc1 -std=c++20 -verify %s

namespace GH57347 {
namespace A {}

void f() {
  using enum A::+; // expected-error {{expected identifier}}
  using enum; // expected-error {{expected identifier or '{'}}
  using enum class; // expected-error {{expected identifier or '{'}}
  using enum : blah; // expected-error {{unknown type name 'blah'}} expected-error {{unnamed enumeration must be a definition}}
}
}
