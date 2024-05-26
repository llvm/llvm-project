// RUN: %clang_cc1 -verify -std=c++20 -fsyntax-only %s
// expected-no-diagnostics

template <typename> struct t1 {
  template <typename>
  struct t2 {};
};

template <typename T>
t1<T>::template t2<T> f1();

void f2() {
  f1<bool>();
}
