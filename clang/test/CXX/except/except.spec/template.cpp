// RUN: %clang_cc1 -std=c++11 -fexceptions -fcxx-exceptions -fsyntax-only -verify %s

// We use pointer assignment compatibility to test instantiation.

template <int N> void f1() throw(int);
template <int N> void f2() noexcept(N > 1);

void (*t1)() throw(int) = &f1<0>;
void (*t2)() throw() = &f1<0>; // expected-error {{not superset}}

void (*t3)() noexcept = &f2<2>; // no-error
void (*t4)() noexcept = &f2<0>; // expected-error {{not superset}}
