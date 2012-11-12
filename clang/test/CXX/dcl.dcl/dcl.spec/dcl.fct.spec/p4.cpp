// RUN: %clang_cc1 -verify %s
// XFAIL: *

void f0() {
}

inline void f0(); // expected-error {{function definition cannot precede inline declaration}}
