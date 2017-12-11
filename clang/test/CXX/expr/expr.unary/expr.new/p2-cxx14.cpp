// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++14 -pedantic

void f() {
  new auto('a');
  new auto {2}; // expected-warning {{ISO C++ standards before C++17 do not allow new expression for type 'auto' to use list-initialization}}
  new auto {1, 2}; // expected-error{{new expression for type 'auto' contains multiple constructor arguments}}
  new auto {}; // expected-error{{new expression for type 'auto' requires a constructor argument}}
  new decltype(auto)({1}); // expected-warning {{ISO C++ standards before C++17 do not allow new expression for type 'decltype(auto)' to use list-initialization}}
  new decltype(auto)({1, 2}); // expected-error{{new expression for type 'decltype(auto)' contains multiple constructor arguments}}
}
