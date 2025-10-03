// RUN: %clang_cc1 -fsyntax-only -std=c++20 -ffp-exception-behavior=strict -verify %s
// expected-no-diagnostics

template <class T> struct S {
  template <class U> using type1 = decltype([] { return U{}; });
};

void foo() {
  using T1 = S<int>::type1<int>;
  int x = T1()();
}
