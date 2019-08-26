// RUN: %clang_cc1 -verify %s
// REQUIRES: thread_support

// expected-warning@* 0-1{{stack nearly exhausted}}
// expected-note@* 0+{{}}

template<int N> struct X : X<N-1> {};
template<> struct X<0> {};
X<1000> x;

template<typename ...T> struct tuple {};
template<typename ...T> auto f(tuple<T...> t) -> decltype(f(tuple<T...>(t))) {} // expected-error {{exceeded maximum depth}}
void g() { f(tuple<int, int>()); }

int f(X<0>);
template<int N> auto f(X<N>) -> f(X<N-1>());

int k = f(X<1000>());
