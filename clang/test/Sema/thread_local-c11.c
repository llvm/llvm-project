// RUN: %clang_cc1 -fsyntax-only -std=c11 -verify %s

void test(void) {
  _Thread_local int x; // expected-error {{'_Thread_local' at block scope requires 'static' or 'extern'}}
  static _Thread_local int y;
  extern _Thread_local int z;
}
