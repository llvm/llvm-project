// RUN: %clang_cc1 -fsyntax-only -verify %s

// If T is the name of a class, then each of the following shall have
// a name different from T:

// - every static data member of class T;
struct X0 {
  static int X0; // expected-error{{member 'X0' has the same name as its class}}
};

// - every member function of class T
// (Cannot be tested)

// - every member of class T that is itself a type;
struct X1 { // expected-note{{previous use is here}}
  enum X1 { }; // expected-error{{use of 'X1' with tag type that does not match previous declaration}}
};

struct X2 {
  typedef int X2; // expected-error{{member 'X2' has the same name as its class}}
};

// - every enumerator of every member of class T that is an enumerated type; and
struct X3 {
  enum E {
    X3 // expected-error{{member 'X3' has the same name as its class}}
  };
};

// - every member of every anonymous union that is a member of class T.
struct X4 {
  union {
    int X;
    union {
      float Y;
      unsigned X4; // expected-error{{member 'X4' has the same name as its class}}
    };
  };
};

