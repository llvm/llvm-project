// RUN: %clang_cc1 -std=c++20 -fsyntax-only %s -verify

struct S {
  operator int();

  template <typename T>
  operator T();
};


// Ensure that no assertion is raised when overload resolution fails while
// choosing between an operator function template and an operator function.
constexpr auto r = &S::operator int;
// expected-error@-1 {{initializer of type '<overloaded function type>'}}
