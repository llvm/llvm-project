// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++14
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++17 -pedantic

void f() {
  new auto('a');
  new auto {2};
  new auto {1, 2}; // expected-error{{new expression for type 'auto' contains multiple constructor arguments}}
  new auto {}; // expected-error{{new expression for type 'auto' requires a constructor argument}}
  new decltype(auto)({1});
  new decltype(auto)({1, 2}); // expected-error{{new expression for type 'decltype(auto)' contains multiple constructor arguments}}
}
