// RUN: %clang_cc1 -std=c1x -fsyntax-only -verify %s

void foo(void) {
  _Generic; // expected-error {{expected '('}}
  (void) _Generic(0); // expected-error {{expected ','}}
  (void) _Generic(0, void); // expected-error {{expected ':'}}
  (void) _Generic(0,
      default: 0,  // expected-note {{previous default generic association is here}}
      default: 0); // expected-error {{duplicate default generic association}}
}

enum E { e };
int bar(int n) {
  // PR45726
  return _Generic(0, enum E: n, default: 0);
}

int baz(int n) {
  // PR39979
  return _Generic(0, enum { e }: n, default: 0);
}
