// RUN: %clang_cc1 -fsyntax-only -verify %s

// In addition, if class T has a user-declared constructor (12.1),
// every non-static data member of class T shall have a name different
// from T.

struct X0 {
  int X0; // okay
};

struct X1 {
  int X1; // expected-error{{member 'X1' has the same name as its class}}
  X1();
};

struct X2 {
  X2();
  float X2; // expected-error{{member 'X2' has the same name as its class}}
};
