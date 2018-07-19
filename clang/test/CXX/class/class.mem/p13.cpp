// RUN: %clang_cc1 -fsyntax-only -verify -std=c++14 %s

// If T is the name of a class, then each of the following shall have
// a name different from T:

// - every static data member of class T;
struct X0 {
  static int X0; // expected-error{{member 'X0' has the same name as its class}}
};

// - every member function of class T
struct Xa {
  int Xa() {} // expected-error{{constructor cannot have a return type}}
};

// - every member of class T that is itself a type;
struct X1 {
  enum X1 { }; // expected-error{{member 'X1' has the same name as its class}}
};

struct X1a {
  struct X1a; // expected-error{{member 'X1a' has the same name as its class}}
};

struct X2 {
  typedef int X2; // expected-error{{member 'X2' has the same name as its class}}
};

struct X2a {
  using X2a = int; // expected-error{{member 'X2a' has the same name as its class}}
};

// - every member template of class T

struct X2b {
  template<typename T> struct X2b; // expected-error{{member 'X2b' has the same name as its class}}
};
struct X2c {
  template<typename T> void X2c(); // expected-error{{constructor cannot have a return type}}
};
struct X2d {
  template<typename T> static int X2d; // expected-error{{member 'X2d' has the same name as its class}}
};
struct X2e {
  template<typename T> using X2e = int; // expected-error{{member 'X2e' has the same name as its class}}
};

// - every enumerator of every member of class T that is an unscoped enumerated type; and
struct X3 {
  enum E {
    X3 // expected-error{{member 'X3' has the same name as its class}}
  };
};
struct X3a {
  enum class E {
    X3a // ok
  };
};

// - every member of every anonymous union that is a member of class T.
struct X4 { // expected-note{{previous}}
  union {
    int X;
    union {
      float Y;
      unsigned X4; // expected-error{{redeclares 'X4'}}
    };
  };
};
