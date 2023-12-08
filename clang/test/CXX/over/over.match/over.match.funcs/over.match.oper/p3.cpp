// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

namespace bullet2 {

// For non-member candidates, if no operand has a class type, only those
// non-member functions that have a matching enumeration parameter are
// candidates.

struct B { template<typename T> B(T); };
int operator~(B);
template<typename T> int operator%(B, T);
enum class E { e };

template<typename T> int f(T t) { return ~t; } // expected-error {{invalid argument type}}
template<typename T, typename U> int f(T t, U u) { return t % u; } // expected-error {{invalid operands to}}

int b1 = ~E::e; // expected-error {{invalid argument type}}
int b2 = f(E::e); // expected-note {{in instantiation of}}
int b3 = f(0, E::e);
int b4 = f(E::e, 0); // expected-note {{in instantiation of}}

}

namespace bullet3 {

// This is specifically testing the bullet:
// "do not have the same parameter-type-list as any non-template
// non-member candidate."
// The rest is sort of hard to test separately.

enum E1 { one };
enum E2 { two };

struct A;

A operator >= (E1, E1);
A operator >= (E1, const E2);

E1 a;
E2 b;

extern A test1;
extern decltype(a >= a) test1;
extern decltype(a >= b) test1;

template <typename T> A operator <= (E1, T);
extern bool test2;
extern decltype(a <= a) test2;

extern A test3;
extern decltype(a <= b) test3;

}
