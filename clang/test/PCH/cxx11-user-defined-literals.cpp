// RUN: %clang_cc1 -pedantic-errors -std=c++11 -emit-pch %s -o %t
// RUN: %clang_cc1 -pedantic-errors -std=c++11 -include-pch %t -verify %s

#ifndef HEADER_INCLUDED

#define HEADER_INCLUDED

using size_t = decltype(sizeof(int));
int operator"" _foo(const char *p, size_t);

template<typename T> auto f(T t) -> decltype(t + ""_foo) { return 0; }

#else

int j = ""_foo;
int k = f(0);
int *l = f(&k);
struct S {};
int m = f(S()); // expected-error {{no matching}}
                // expected-note@11 {{substitution failure}}

#endif
