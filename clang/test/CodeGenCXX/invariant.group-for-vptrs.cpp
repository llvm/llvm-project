// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -emit-llvm %s -fstrict-vtable-pointers -O1 -o - -disable-llvm-passes | FileCheck %s

struct A {
  virtual void foo();
};

struct D : A {
  void foo();
};

// CHECK-LABEL: define void @_Z21testExternallyVisiblev()
void testExternallyVisible() {
  A *a = new A;

  // CHECK: load {{.*}} !invariant.group ![[A_MD:[0-9]+]]
  a->foo();

  D *d = new D;
  // CHECK: call void @_ZN1DC1Ev(
  // CHECK: load {{.*}} !invariant.group ![[D_MD:[0-9]+]]
  d->foo();
  A *a2 = d;
  // CHECK: load {{.*}} !invariant.group ![[A_MD]]
  a2->foo();
}
// CHECK-LABEL: {{^}}}

namespace {

struct B {
  virtual void bar();
};

struct C : B {
  void bar();
};

}

// CHECK-LABEL: define void @_Z21testInternallyVisibleb(
void testInternallyVisible(bool p) {
  B *b = new B;
  // CHECK: = load {{.*}}, !invariant.group ![[B_MD:[0-9]+]]
  b->bar();

  // CHECK: call void @_ZN12_GLOBAL__N_11CC1Ev(
  C *c = new C;
  // CHECK: = load {{.*}}, !invariant.group ![[C_MD:[0-9]+]]
  c->bar();
}

// Checking A::A()
// CHECK-LABEL: define linkonce_odr void @_ZN1AC2Ev(
// CHECK: store {{.*}}, !invariant.group ![[A_MD]]
// CHECK-LABEL: {{^}}}

// Checking D::D()
// CHECK-LABEL: define linkonce_odr void @_ZN1DC2Ev(
// CHECK:  = call i8* @llvm.invariant.group.barrier(i8*
// CHECK:  call void @_ZN1AC2Ev(%struct.A*
// CHECK: store {{.*}} !invariant.group ![[D_MD]]

// Checking B::B()
// CHECK-LABEL: define internal void @_ZN12_GLOBAL__N_11BC2Ev(
// CHECK:  store {{.*}}, !invariant.group ![[B_MD]]

// Checking C::C()
// CHECK-LABEL: define internal void @_ZN12_GLOBAL__N_11CC2Ev(
// CHECK:  store {{.*}}, !invariant.group ![[C_MD]]

// CHECK: ![[A_MD]] = !{!"_ZTS1A"}
// CHECK: ![[D_MD]] = !{!"_ZTS1D"}
// CHECK: ![[B_MD]] = distinct !{}
// CHECK: ![[C_MD]] = distinct !{}
