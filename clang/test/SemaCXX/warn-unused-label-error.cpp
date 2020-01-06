// RUN: %clang_cc1 -fsyntax-only -Wunused-variable -Wunused-label -verify %s

static int unused_local_static;

namespace PR8455 {
  void f() {
    A: // expected-warning {{unused label 'A'}}
      __attribute__((unused)) int i; // attribute applies to variable
    B: // attribute applies to label
      __attribute__((unused)); int j; // expected-warning {{unused variable 'j'}}
  }

  void g() {
    C: // unused label 'C' will not appear here because an error has occurred
      __attribute__((unused))
      #pragma weak unused_local_static  // expected-error {{expected ';' after __attribute__}}
      ;
  }

  void h() {
    D: // expected-warning {{unused label 'D'}}
      #pragma weak unused_local_static
      __attribute__((unused))  // expected-warning {{declaration does not declare anything}}
      ;
  }
}
