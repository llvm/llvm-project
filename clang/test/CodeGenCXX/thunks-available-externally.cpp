// RUN: %clang_cc1 %s -I%S -triple=x86_64-apple-darwin10 -emit-llvm -O3 -o - | FileCheck %s

// Check that we don't assert on this case.
namespace Test1 {

struct Incomplete;

struct A {
  virtual void f();
  virtual void g(Incomplete);
  virtual void h();
  virtual void i();
  int a;
};

struct B {
  virtual void f();
  virtual void g(Incomplete);
  virtual void h();
  virtual void i();
  int b;
};

struct C : A, B {
  C();

  virtual void f();
  virtual void g(Incomplete);
  virtual void h();
  virtual void i();
};

void C::h() { }

C::C() { }

void C::i() { }

}

namespace Test2 {

struct A {
  virtual void f();
  int a;
};

struct B {
  virtual void f();
  int b;
};

struct C : A, B {
  virtual void f();
};

static void f(B* b) {
  b->f();
}

// CHECK: define void @_ZN5Test21fEv()
// CHECK: call void @_ZN5Test21C1fEv
// CHECK: ret void
// CHECK: define available_externally void @_ZThn16_N5Test21C1fEv
void f() {
  C c;
  f(&c);
}

}

// Test that we don't assert.
namespace Test3 {

struct A {
  virtual ~A();

  int a;
};

struct B : A { };
struct C : virtual B { };

void f() {
  C c;
}

}
