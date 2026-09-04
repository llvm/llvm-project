// Test that Clang emits vtable metadata when speculative devirtualization is enabled.
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fdevirtualize-speculatively -emit-llvm -o - %s | FileCheck --check-prefix=CHECK  %s

struct A {
  A();
  virtual void f();
};

struct B : virtual A {
  B();
  virtual void g();
  virtual void h();
};

namespace {

struct D : B {
  D();
  virtual void f();
  virtual void h();
};

}

A::A() {}
B::B() {}
D::D() {}

void A::f() {
}

void B::g() {
}

void D::f() {
}

void D::h() {
}

void af(A *a) {
  // CHECK: [[P:%[^ ]*]] = call i1 @llvm.public.type.test(ptr [[VT:%[^ ]*]], metadata !"_ZTS1A")
  // CHECK-NEXT: call void @llvm.assume(i1 [[P]])
  a->f();
}

void dg1(D *d) {
  // CHECK: [[P:%[^ ]*]] = call i1 @llvm.public.type.test(ptr [[VT:%[^ ]*]], metadata !"_ZTS1B")
  // CHECK-NEXT: call void @llvm.assume(i1 [[P]])
  d->g();
}

void df1(D *d) {
  // CHECK: [[P:%[^ ]*]] = call i1 @llvm.type.test(ptr [[VT:%[^ ]*]], metadata !11)
  // CHECK-NEXT: call void @llvm.assume(i1 [[P]])
  d->f();
}

void dh1(D *d) {
  // CHECK: [[P:%[^ ]*]] = call i1 @llvm.type.test(ptr [[VT:%[^ ]*]], metadata !11)
  // CHECK-NEXT: call void @llvm.assume(i1 [[P]])
  d->h();
}


D d;

void foo() {
  dg1(&d);
  df1(&d);
  dh1(&d);


  struct FA : A {
    void f() {}
  } fa;
  af(&fa);
}
