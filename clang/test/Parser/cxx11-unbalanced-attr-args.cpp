// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,pedantic -std=c++11 -pedantic %s
// RUN: %clang_cc1 -fsyntax-only -verify -x c -std=c23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,pedantic -x c -std=c23 -pedantic %s

void test() {
  [[X1(])]]; // expected-warning {{unknown attribute 'X1' ignored}} pedantic-warning {{attribute argument list is not a balanced token sequence}}
  [[X1(})]]; // expected-warning {{unknown attribute 'X1' ignored}} pedantic-warning {{attribute argument list is not a balanced token sequence}}
  [[X1]];    // expected-warning {{unknown attribute 'X1' ignored}}
  [[X1()]];  // expected-warning {{unknown attribute 'X1' ignored}}
}
