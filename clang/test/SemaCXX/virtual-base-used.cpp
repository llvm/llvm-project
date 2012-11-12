// RUN: %clang_cc1 -fsyntax-only -verify %s
// PR7800

class NoDestroy { ~NoDestroy(); }; // expected-note 3 {{declared private here}}
struct A {
  virtual ~A();
};

struct B : public virtual A {
  NoDestroy x; // expected-error {{field of type 'NoDestroy' has private destructor}}
};
struct D : public virtual B {
  virtual void foo();
  ~D();
};
void D::foo() { // expected-note {{implicit default destructor for 'B' first required here}}
}

struct E : public virtual A {
  NoDestroy x; // expected-error {{field of type 'NoDestroy' has private destructor}}
};
struct F : public E { // expected-note {{implicit default destructor for 'E' first required here}}
};
struct G : public virtual F {
  virtual void foo();
  ~G();
};
void G::foo() { // expected-note {{implicit default destructor for 'F' first required here}}
}

struct H : public virtual A {
  NoDestroy x; // expected-error {{field of type 'NoDestroy' has private destructor}}
};
struct I : public virtual H {
  ~I();
};
struct J : public I {
  virtual void foo();
  ~J();
};
void J::foo() { // expected-note {{implicit default destructor for 'H' first required here}}
}
