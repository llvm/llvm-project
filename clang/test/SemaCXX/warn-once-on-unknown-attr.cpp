// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
// RUN: %clang_cc1 -fsyntax-only -verify -std=c2x -x c %s
void foo() {
  int [[attr]] i;             // expected-warning {{unknown attribute 'attr' ignored}}
  (void)sizeof(int [[attr]]); // expected-warning {{unknown attribute 'attr' ignored}}
}

void bar() {
  [[attr]];       // expected-warning {{unknown attribute 'attr' ignored}}
  [[attr]] int i; // expected-warning {{unknown attribute 'attr' ignored}}
}
