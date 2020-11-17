// RUN: %clang_cc1 -std=c++98 %s -triple armv7-none-eabi -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++11 %s -triple armv7-none-eabi -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++1z %s -triple armv7-none-eabi -emit-llvm -o - | FileCheck %s

struct A {
  virtual void f();
  virtual void f_const() const;
  virtual void g();

  A h();
};

A g();

void f(A a, A *ap, A& ar) {
  // This should not be a virtual function call.
  
  // CHECK: call void @_ZN1A1fEv(%struct.A* {{[^,]*}} %a)
  a.f();

  // CHECK: call void %  
  ap->f();

  // CHECK: call void %  
  ar.f();
  
  // CHECK: call void @_ZN1A1fEv
  A().f();

  // CHECK: call void @_ZN1A1fEv
  g().f();
  
  // CHECK: call void @_ZN1A1fEv
  a.h().f();

  // CHECK: call void @_ZNK1A7f_constEv
  a.f_const();

  // CHECK: call void @_ZN1A1fEv
  (a).f();
}

struct D : A { virtual void g(); };
struct XD { D d; };

D gd();

void fd(D d, XD xd, D *p) {
  // CHECK: call void @_ZN1A1fEv(%struct.A*
  d.f();

  // CHECK: call void @_ZN1D1gEv(%struct.D*
  d.g();

  // CHECK: call void @_ZN1A1fEv
  D().f();

  // CHECK: call void @_ZN1D1gEv
  D().g();

  // CHECK: call void @_ZN1A1fEv
  gd().f();
  
  // CHECK: call void @_ZNK1A7f_constEv
  d.f_const();

  // CHECK: call void @_ZN1A1fEv
  (d).f();

  // CHECK: call void @_ZN1A1fEv
  (true, d).f();

  // CHECK: call void @_ZN1D1gEv
  (true, d).g();

  // CHECK: call void @_ZN1A1fEv
  xd.d.f();

  // CHECK: call void @_ZN1A1fEv
  XD().d.f();

  // CHECK: call void @_ZN1A1fEv
  D XD::*mp;
  (xd.*mp).f();

  // CHECK: call void @_ZN1D1gEv
  (xd.*mp).g();

  // Can't devirtualize this; we have no guarantee that p points to a D here,
  // due to the "single object is considered to be an array of one element"
  // rule.
  // CHECK: call void %
  p[0].f();

  // FIXME: We can devirtualize this, by C++1z [expr.add]/6 (if the array
  // element type and the pointee type are not similar, behavior is undefined).
  // CHECK: call void %
  p[1].f();
}

struct B {
  virtual void f();
  ~B();
  
  B h();
};


void f() {
  // CHECK: call void @_ZN1B1fEv
  B().f();
  
  // CHECK: call void @_ZN1B1fEv
  B().h().f();
}

namespace test2 {
  struct foo {
    virtual void f();
    virtual ~foo();
  };

  struct bar : public foo {
    virtual void f();
    virtual ~bar();
  };

  void f(bar *b) {
    // CHECK: call void @_ZN5test23foo1fEv
    // CHECK: call %"struct.test2::foo"* @_ZN5test23fooD1Ev
    b->foo::f();
    b->foo::~foo();
  }
}

namespace test3 {
  // Test that we don't crash in this case.
  struct B {
  };
  struct D : public B {
  };
  void f(D d) {
    // CHECK-LABEL: define void @_ZN5test31fENS_1DE
    d.B::~B();
  }
}

namespace test4 {
  struct Animal {
    virtual void eat();
  };
  struct Fish : Animal {
    virtual void eat();
  };
  struct Wrapper {
    Fish fish;
  };
  extern Wrapper *p;
  void test() {
    // CHECK: call void @_ZN5test44Fish3eatEv
    p->fish.eat();
  }
}

// Do not devirtualize to pure virtual function calls.
namespace test5 {
  struct X {
    virtual void f() = 0;
  };
  struct Y {};
  // CHECK-LABEL: define {{.*}} @_ZN5test51f
  void f(Y &y, X Y::*p) {
    // CHECK-NOT: call {{.*}} @_ZN5test51X1fEv
    // CHECK: call void %
    (y.*p).f();
  };

  struct Z final {
    virtual void f() = 0;
  };
  // CHECK-LABEL: define {{.*}} @_ZN5test51g
  void g(Z &z) {
    // CHECK-NOT: call {{.*}} @_ZN5test51Z1fEv
    // CHECK: call void %
    z.f();
  }

  struct Q {
    virtual void f() final = 0;
  };
  // CHECK-LABEL: define {{.*}} @_ZN5test51h
  void h(Q &q) {
    // CHECK-NOT: call {{.*}} @_ZN5test51Q1fEv
    // CHECK: call void %
    q.f();
  }
}
