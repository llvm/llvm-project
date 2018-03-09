// RUN: %clangxx_cfi -o %t %s
// RUN: %run %t

// Tests that the CFI mechanism does not crash the program when making various
// kinds of valid calls involving classes with various different linkages and
// types of inheritance, and both virtual and non-virtual member functions.

#include "utils.h"

struct A {
  virtual void f();
  void g();
};

void A::f() {}
void A::g() {}

struct A2 : A {
  virtual void f();
  void g();
};

void A2::f() {}
void A2::g() {}

struct B {
  virtual void f() {}
  void g() {}
};

struct B2 : B {
  virtual void f() {}
  void g() {}
};

namespace {

struct C {
  virtual void f();
  void g();
};

void C::f() {}
void C::g() {}

struct C2 : C {
  virtual void f();
  void g();
};

void C2::f() {}
void C2::g() {}

struct D {
  virtual void f() {}
  void g() {}
};

struct D2 : D {
  virtual void f() {}
  void g() {}
};

}

struct E {
  virtual void f() {}
  void g() {}
};

struct E2 : virtual E {
  virtual void f() {}
  void g() {}
};

int main() {
  A *a = new A;
  break_optimization(a);
  a->f();
  a->g();
  a = new A2;
  break_optimization(a);
  a->f();
  a->g();

  B *b = new B;
  break_optimization(b);
  b->f();
  b->g();
  b = new B2;
  break_optimization(b);
  b->f();
  b->g();

  C *c = new C;
  break_optimization(c);
  c->f();
  c->g();
  c = new C2;
  break_optimization(c);
  c->f();
  c->g();

  D *d = new D;
  break_optimization(d);
  d->f();
  d->g();
  d = new D2;
  break_optimization(d);
  d->f();
  d->g();

  E *e = new E;
  break_optimization(e);
  e->f();
  e->g();
  e = new E2;
  break_optimization(e);
  e->f();
  e->g();
}
