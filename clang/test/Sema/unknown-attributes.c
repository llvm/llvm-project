// RUN: %clang_cc1 -Wunknown-attributes -fsyntax-only -verify %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wunknown-attributes -verify %s

[[foo::a]] // expected-warning {{unknown attribute 'foo::a' ignored}}
int f1(void) {
  return 0;
}

[[clan::deprecated]] // expected-warning {{unknown attribute 'clan::deprecated' ignored}}
int f2(void) {
  return 0;
}
