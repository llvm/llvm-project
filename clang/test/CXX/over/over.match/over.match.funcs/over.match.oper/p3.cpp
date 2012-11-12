// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

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