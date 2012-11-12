// RUN: %clang_cc1 -fsyntax-only -verify %s
struct A {
  virtual void f() = 0; // expected-note 2{{overridden virtual function}}
};

struct Aprime : virtual A {
  virtual void f();
};

struct B : Aprime {
  virtual void f(); // expected-note 3{{final overrider of 'A::f'}}
};

struct C : virtual A {
  virtual void f(); // expected-note{{final overrider of 'A::f'}}
};

struct D : B, C { }; // expected-error{{virtual function 'A::f' has more than one final overrider in 'D'}}

struct B2 : B { };

struct E : B, B2 { }; //expected-error{{virtual function 'A::f' has more than one final overrider in 'E'}}

struct F : B, B2 {
  virtual void f(); // okay
};

struct G : F { }; // okay

struct H : G, A { }; // okay

namespace MultipleSubobjects {
  struct A { virtual void f(); };
  struct B : A { virtual void f(); };
  struct C : A { virtual void f(); };
  struct D : B, C { }; // okay
}
