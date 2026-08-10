// RUN: %clang_cc1 %s -verify -fsyntax-only
// PR99933

struct S {
  template <typename> int i; // expected-error {{non-static data member 'i' cannot be declared as a template}}
};
