// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

template<typename T=int> struct S {};
template<typename> void f();

template<typename T, typename... V> struct S<T(V...)> {};

template<typename ...T> struct V {};
template<typename ...T> struct V<void(T)...> {};

void foo(void) {
  // In C++11 mode, all of these are expected to parse correctly, and the CUDA
  // language should not interfere with that.

  // expected-no-diagnostics

  S<S<S<int>>> s3;
  S<S<S<>>> s30;

  S<S<S<S<int>>>> s4;
  S<S<S<S<>>>> s40;

  S<S<S<S<S<int>>>>> s5;
  S<S<S<S<S<>>>>> s50;

  (void)(&f<S<S<int>>>==0);
  (void)(&f<S<S<>>>==0);

  S<S<S<void()>>> s6;
}

template<typename ...T>
void bar(T... args) {
  S<S<V<void(T)...>>> s7;
}
