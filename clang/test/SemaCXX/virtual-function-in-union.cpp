// RUN: %clang_cc1 -fsyntax-only -verify %s

union U {
  int d;
  virtual int f() { return d; }; // expected-error {{unions cannot have virtual functions}}
};

int foo() { U u; return u.d; }
