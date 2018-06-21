// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr777 { // dr777: 3.7
#if __cplusplus >= 201103L
template <typename... T>
void f(int i = 0, T ...args) {}
void ff() { f(); }

template <typename... T>
void g(int i = 0, T ...args, T ...args2) {}

template <typename... T>
void h(int i = 0, T ...args, int j = 1) {}
#endif
}

// expected-no-diagnostics
