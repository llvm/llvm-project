// RUN: %clang_cc1 -std=c1x -fsyntax-only -verify %s

void foo(void) {
  _Generic; // expected-error {{expected '('}}
  (void) _Generic(0); // expected-error {{expected ','}}
  (void) _Generic(0, void); // expected-error {{expected ':'}}
  (void) _Generic(0,
      default: 0,  // expected-note {{previous default generic association is here}}
      default: 0); // expected-error {{duplicate default generic association}}
}
