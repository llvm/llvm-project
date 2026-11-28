// RUN: %clang_cc1 -fsyntax-only -std=c++20 -Wpre-c++20-compat-pedantic -verify %s

namespace GH161702 {
struct S {
  enum E { A };
  using E::A; // expected-warning {{member using declaration naming non-class ''E'' enumerator is incompatible with C++ standards before C++20}}
};
}
