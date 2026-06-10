// RUN: %clang_cc1 -std=c++11 -verify %s

template<typename ...T> struct X {
  void f(int);
  void f(...);
  static int n;
};

template<typename T, typename U> using A = T;

// FIXME: These definitions are not OK, X<A<T, decltype(...)>...> is not equivalent to X<T...>.
template<typename ...T>
void X<A<T, decltype(f(T()))>...>::f(int) {}

template<typename ...T>
int X<A<T, decltype(f(T()))>...>::n = 0; // expected-error {{undeclared}}

struct Y {}; void f(Y);

void g() {
  // OK, substitution succeeds.
  X<Y>().f(0);
  X<Y>::n = 1;

  // FIXME: There should be no substitutiton failure since the out-of-line definitions were not valid.
  X<void>().f(0);
  X<void>::n = 1; // expected-note {{instantiation of}}
}
