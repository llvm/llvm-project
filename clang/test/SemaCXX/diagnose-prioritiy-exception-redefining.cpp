// RUN: %clang_cc1 -fsyntax-only -verify -fcxx-exceptions -std=c++11 %s

struct ExplicitlySpecialMethod {
  ExplicitlySpecialMethod() = default;
};
ExplicitlySpecialMethod::ExplicitlySpecialMethod() {} // expected-error{{definition of explicitly defaulted default constructor}}

struct ImplicitlySpecialMethod {};
ImplicitlySpecialMethod::ImplicitlySpecialMethod() {} // expected-error{{definition of implicitly declared default constructor}}
