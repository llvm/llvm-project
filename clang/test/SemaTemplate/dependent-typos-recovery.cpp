// RUN: %clang_cc1 -fsyntax-only -verify %s

// There should be no extra errors about missing 'template' keywords.
struct B {
  template <typename T>
  int f(){};
} builder;                // expected-note 2{{'builder' declared here}}

auto a = bilder.f<int>(); // expected-error{{undeclared identifier 'bilder'; did you mean}}
auto b = (*(&bilder+0)).f<int>(); // expected-error{{undeclared identifier 'bilder'; did you mean}}
