// RUN: %clang_cc1 -std=c++98 -fgnu-keywords -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++11 -fgnu-keywords -fsyntax-only -verify %s

struct B {
  virtual void g();
};
struct D __final : B { // expected-warning {{__final is a GNU extension, consider using C++11 final}}
  virtual void g() __final; // expected-warning {{__final is a GNU extension, consider using C++11 final}}
};
