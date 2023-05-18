// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
// expected-no-diagnostics

struct S {};
int x;
S&& y1 = (S&&)x;
S&& y2 = reinterpret_cast<S&&>(x);
S& z1 = (S&)x;
