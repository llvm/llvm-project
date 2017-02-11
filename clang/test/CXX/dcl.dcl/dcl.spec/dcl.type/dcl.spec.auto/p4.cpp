// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++1y
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11 -Wno-c++1y-extensions
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++98 -Wno-c++11-extensions

template<typename T>
struct only {
  only(T);
  template<typename U> only(U) = delete;
};

void f() {
  if (auto a = true) {
  }

  switch (auto a = 0) {
  }

  while (auto a = false) {
  }

  for (; auto a = false; ) {
  }

  new const auto (0);
  new (auto) (0.0);

  int arr[] = {1, 2, 3};
  for (auto i : arr) {
  }
}

class X {
  static const auto n = 'x';

  auto m = 0; // expected-error {{'auto' not allowed in non-static class member}}
};

struct S {
  static const auto a; // expected-error {{declaration of variable 'a' with deduced type 'const auto' requires an initializer}}
  static const auto b = 0;
  static const int c;
};
const int S::b;
const auto S::c = 0;

namespace std { template<typename T> struct initializer_list { initializer_list(); }; }

// In an initializer of the form ( expression-list ), the expression-list
// shall be a single assigment-expression.
auto parens1(1);
auto parens2(2, 3); // expected-error {{initializer for variable 'parens2' with type 'auto' contains multiple expressions}}
#if __cplusplus >= 201103L
auto parens3({4, 5, 6}); // expected-error {{cannot deduce type for variable 'parens3' with type 'auto' from parenthesized initializer list}}
auto parens4 = [p4(1)] {};
auto parens5 = [p5(2, 3)] {}; // expected-error {{initializer for lambda capture 'p5' contains multiple expressions}}
auto parens6 = [p6({4, 5, 6})] {}; // expected-error {{cannot deduce type for lambda capture 'p6' from parenthesized initializer list}}
#endif

#if __cplusplus >= 201402L
namespace std_example {
  // The other half of this example is in p3.cpp
  auto f() -> int;
  auto g() { return 0.0; }
  auto h();
}
#endif
