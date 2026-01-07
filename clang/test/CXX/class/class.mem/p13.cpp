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

// This includes such things inherited from base classes.
struct B {
  static int D0;
  int Da() {};
  enum D1 {};
  struct D1a;
  typedef int D2;
  using D2a = int;
  template<typename T> struct D2b;
  template<typename T> void D2c();
  template<typename T> static int D2d;
  template<typename T> using D2e = int;
  union { int D4; };
  int Dtemplate;
  int Dtemplate_with_ctors;
};
struct B2 { int Dtemplate(); };

struct D0 : B { using B::D0; }; // expected-error {{member 'D0' has the same name as its class}}
struct Da : B { using B::Da; }; // expected-error {{member 'Da' has the same name as its class}}
struct D1 : B { using B::D1; }; // expected-error {{member 'D1' has the same name as its class}}
struct D1a : B { using B::D1a; }; // expected-error {{member 'D1a' has the same name as its class}}
struct D2 : B { using B::D2; }; // expected-error {{member 'D2' has the same name as its class}}
struct D2a : B { using B::D2a; }; // expected-error {{member 'D2a' has the same name as its class}}
struct D2b : B { using B::D2b; }; // expected-error {{member 'D2b' has the same name as its class}}
struct D2c : B { using B::D2c; }; // expected-error {{member 'D2c' has the same name as its class}}
struct D2d : B { using B::D2d; }; // expected-error {{member 'D2d' has the same name as its class}}
struct D2e : B { using B::D2e; }; // expected-error {{member 'D2e' has the same name as its class}}
struct D4 : B { using B::D4; }; // expected-error {{member 'D4' has the same name as its class}}

template<typename B> struct Dtemplate : B {
  using B::Dtemplate; // expected-error {{member 'Dtemplate' has the same name as its class}}
};
Dtemplate<B> ok;
Dtemplate<B2> error; // expected-note {{in instantiation of}}

template<typename B> struct Dtemplate_with_ctors : B {
  Dtemplate_with_ctors();
  using B::Dtemplate_with_ctors; // expected-error {{member 'Dtemplate_with_ctors' has the same name as its class}}
};

template<typename B> struct CtorDtorName : B {
  using B::CtorDtorName; // expected-error {{member 'CtorDtorName' has the same name as its class}} expected-note {{non-type declaration found by destructor name lookup}}
  CtorDtorName();
  ~CtorDtorName(); // expected-error {{identifier 'CtorDtorName' after '~' in destructor name does not name a type}}
};

struct S { // expected-note {{'S' declared here}}
  enum E {
    R = 11,
    S = 12  // expected-error {{member 'S' has the same name as its class}}
  };
  static_assert(E::R == 11, "E::R is not 11");
  static_assert(E::S == 12, "E::S is not 12");  // expected-error {{no member named 'S' in 'S::E'}}
};
