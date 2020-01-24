// RUN:  %clang_cc1 -std=c++2a -verify %s

// Make sure constraint expressions are unevaluated before being substituted
// into during satisfaction checking.

template<typename T> constexpr int f() { return T::value; }
template<typename T> concept Foo = false && (f<int>(), true);
bool k = Foo<int>;
template<typename T> requires false && (f<int>(), true) struct S {};
// expected-note@-1{{because}}
using s = S<int>; // expected-error {{constraints not satisfied}}
template<typename T> void foo() requires false && (f<int>(), true) { };
// expected-note@-1{{because}} expected-note@-1{{candidate template ignored}}
int a = (foo<int>(), 0); // expected-error{{no matching function}}
template<typename T> void bar() requires requires { requires false && (f<int>(), true); } { };
// expected-note@-1{{because}} expected-note@-1{{candidate template ignored}}
int b = (bar<int>(), 0); // expected-error{{no matching function}}