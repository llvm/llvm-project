// RUN: seq 100 | xargs -Ifoo %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics
// This is a regression test for a non-deterministic stack-overflow.

template < typename >
concept C1 = true;

template < typename , auto >
concept C2 = true;

template < C1 auto V, C2< V > auto>
struct S;
