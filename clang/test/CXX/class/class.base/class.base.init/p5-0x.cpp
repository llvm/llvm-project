// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// [class.base.init]p5
// A ctor-initializer may initialize a variant member of the constructor’s 
// class. If a ctor-initializer specifies more than one mem-initializer for the
// same member or for the same base class, the ctor-initializer is ill-formed.

union E {
  int a;
  int b;
  E() : a(1),  // expected-note{{previous initialization is here}}
        b(2) { // expected-error{{initializing multiple members of union}}
  }
};

union F {
  struct {
    int a;
    int b;
  };
  int c;
  F() : a(1),  // expected-note{{previous initialization is here}}
        b(2),
        c(3) { // expected-error{{initializing multiple members of union}}
  }
};
