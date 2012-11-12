// RUN: %clang_cc1 -fsyntax-only -verify %s
struct A {
  virtual ~A();
};

struct B : A { // expected-error {{no suitable member 'operator delete' in 'B'}}
  virtual void f();

  void operator delete (void *, int); // expected-note {{'operator delete' declared here}}
};

void B::f() { // expected-note {{implicit default destructor for 'B' first required here}}
}

struct C : A { // expected-error {{no suitable member 'operator delete' in 'C'}}
  C();
  void operator delete(void *, int); // expected-note {{'operator delete' declared here}}
};

C::C() { }  // expected-note {{implicit default destructor for 'C' first required here}}

struct D : A { // expected-error {{no suitable member 'operator delete' in 'D'}}
  void operator delete(void *, int); // expected-note {{'operator delete' declared here}}
}; 

void f() {
  new D; // expected-note {{implicit default destructor for 'D' first required here}}
}

