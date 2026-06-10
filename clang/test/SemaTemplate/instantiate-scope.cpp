// RUN: %clang_cc1 -std=c++11 -verify %s

template<typename ...T> struct X {
  void f(int);
  void f(...);
  static int n;
};

template<typename T, typename U> using A = T;

// These definitions are not OK, X<A<T, decltype(...)>...> is not equivalent to X<T...>.
template<typename ...T>
void X<A<T, decltype(f(T()))>...>::f(int) {}
// expected-error@-1 {{nested name specifier 'X<A<T, decltype(f(T()))>...>' for declaration does not refer into a class}}

template<typename ...T>
int X<A<T, decltype(f(T()))>...>::n = 0;
// expected-error@-1 {{nested name specifier 'X<A<T, decltype(f(T()))>...>' for declaration does not refer into a class}}

struct Y {}; void f(Y);

void g() {
  // OK, substitution succeeds.
  X<Y>().f(0);
  X<Y>::n = 1;

  // No substitutiton failure since the out-of-line definitions were not valid.
  X<void>().f(0);
  X<void>::n = 1;
}
