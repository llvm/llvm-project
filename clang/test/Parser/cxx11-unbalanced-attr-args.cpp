// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -x c -std=c23 %s

[[X1(])]]; // expected-error {{expected ')'}} expected-warning {{unknown attribute 'X1' ignored}}
[[X1(})]]; // expected-error {{expected ')'}} expected-warning {{unknown attribute 'X1' ignored}}
[[X1]]; // expected-warning {{unknown attribute 'X1' ignored}}
[[X1()]]; // expected-warning {{unknown attribute 'X1' ignored}}