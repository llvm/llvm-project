// RUN: %clang_cc1 -std=c23 -verify -pedantic %s

const int a = 0;
struct C { const int *p; };
constexpr struct C c1 = {&a}; // expected-error{{constexpr pointer initializer is not null}}
constexpr struct C c2 = (struct C){&a}; // expected-error{{constexpr variable 'c2' must be initialized by a constant expression}}

union U { const int *p; int x; };
constexpr union U u1 = {.p = &a}; // expected-error{{constexpr pointer initializer is not null}}
constexpr union U u2 = (union U){.p = &a}; // expected-error{{constexpr variable 'u2' must be initialized by a constant expression}}
constexpr union U u3 = {.x = 1};
constexpr union U u4 = (union U){.x = 1};
