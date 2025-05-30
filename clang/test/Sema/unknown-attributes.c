// RUN: %clang_cc1 -Wunknown-attributes -fsyntax-only -verify %s
// RUN: %clang_cc1 -x c++ -Wunknown-attributes -fsyntax-only -verify %s

[[gmu::deprected]] // expected-warning {{unknown attribute 'gmu::deprected' ignored; did you mean 'gnu::deprecated'?}}
int f1(void) {
  return 0;
}

[[gmu::deprecated]] // expected-warning {{unknown attribute 'gmu::deprecated' ignored; did you mean 'gnu::deprecated'?}}
int f2(void) {
  return 0;
}

[[gnu::deprected]] // expected-warning {{unknown attribute 'gnu::deprected' ignored; did you mean 'gnu::deprecated'?}}
int f3(void) {
  return 0;
}

[[deprected]] // expected-warning {{unknown attribute 'deprected' ignored; did you mean 'deprecated'?}}
int f4(void) {
  return 0;
}
