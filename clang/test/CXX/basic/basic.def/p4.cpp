// RUN: %clang_cc1 -std=c++1z -verify %s

inline int f(); // expected-warning {{inline function 'f' is not defined}}
extern inline int n; // expected-error {{inline variable 'n' is not defined}}

int use = f() + n; // expected-note 2{{used here}}
