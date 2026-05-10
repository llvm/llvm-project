// RUN: %clang_cc1 -fsyntax-only -verify %s

template <class T> int f(T) {
  struct A {
    static int B; // expected-error {{static data member 'B' not allowed in local struct 'A'}}
  };
  int A::B; // expected-note {{previous definition is here}}
  int A::B = 1; // expected-error {{redefinition of 'B'}}
  return 0;
}

int x = f(0);
