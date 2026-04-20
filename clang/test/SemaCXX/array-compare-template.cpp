// RUN: %clang_cc1 -std=c++26 -pedantic-errors -fsyntax-only -verify %s

void test() {
  int a[1]{}, b[1]{};
  [](const auto &x, const auto &y) {
    return x == y; // expected-error {{comparison between two arrays is ill-formed in C++26}}
  }(a, b); // expected-note {{in instantiation of function template specialization}}
}
