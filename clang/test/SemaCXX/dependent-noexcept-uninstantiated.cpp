// RUN: %clang_cc1 -fsyntax-only -std=c++17 %s
// expected-no-diagnostics

using A = int;
using B = char;

template <class T> struct C {
  template <class V> void f0() noexcept(sizeof(T) == sizeof(A) && sizeof(V) == sizeof(B)) {}
  template <class V> auto f1(V a) noexcept(1) {return a;}
};

void (C<int>::*tmp0)() noexcept = &C<A>::f0<B>;
int (C<int>::*tmp1)(int) noexcept = &C<A>::f1;
