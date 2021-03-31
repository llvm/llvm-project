// RUN: %clang_cc1 -x c++ -fsyntax-only %s -verify

static union {     // expected-warning {{declaration does not declare anything}}
  virtual int a(); // expected-error {{unions cannot have virtual functions}} \
                   // expected-error {{functions cannot be declared in an anonymous union}}
};
