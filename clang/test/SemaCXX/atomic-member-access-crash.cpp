// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// This test verifies that accessing members of an _Atomic struct
// produces diagnostics instead of crashing.

struct S {};
constexpr S s;

struct SS {
  consteval const S *operator->() const { return &s; }
  consteval int foo() const { return 42; }
};

_Atomic SS ss; // expected-note {{declared here}}

int v = ss->bar; // expected-error {{no member named 'bar' in 'SS'}} \
                 // expected-error {{call to consteval function 'SS::operator->' is not a constant expression}} \
                 // expected-note {{read of non-constexpr variable 'ss' is not allowed in a constant expression}}
int x = ss.foo(); // expected-error {{accessing a member of an atomic structure or union is undefined behavior}}
