// RUN: %clang_cc1 -fsyntax-only -verify %s

struct x {
  x() : a(4) ; // expected-error {{expected '{'}}
};

struct y {
  int a;
  y() : a(4) ; // expected-error {{expected '{'}}
};

struct z {
  int a;
  z() : a {} // expected-error {{expected '('}}
};
