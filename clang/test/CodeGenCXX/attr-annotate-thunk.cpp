// RUN: %clang_cc1 %s -emit-llvm -triple x86_64-unknown-linux-gnu -o - | FileCheck %s

// Annotations on a virtual function are deferred to the end of the TU, keyed
// by mangled name. For a this-adjusting thunk that name is mangled into a
// stack buffer in CodeGenVTables::maybeEmitThunk, so the deferred-annotation
// map must own a copy of the key instead of referencing the caller's storage.

struct A {
  virtual void f();
  virtual ~A();
};

struct B {
  virtual void g();
  virtual ~B();
};

struct C : A, B {
  void f() override;
  __attribute__((annotate("annotated_method"))) void g() override;
  __attribute__((annotate("annotated_dtor"))) ~C() override;
};

void C::f() {}
void C::g() {}
C::~C() {}

// Each annotation is recorded for the function itself and for the thunk that
// adjusts `this` to the B subobject.

// CHECK: @[[METHOD:.*]] = private unnamed_addr constant [17 x i8] c"annotated_method\00", section "llvm.metadata"
// CHECK: @[[DTOR:.*]] = private unnamed_addr constant [15 x i8] c"annotated_dtor\00", section "llvm.metadata"
// CHECK: @llvm.global.annotations = appending global [7 x { ptr, ptr, ptr, i32, ptr }] [
// CHECK-SAME: { ptr @_ZN1C1gEv, ptr @[[METHOD]],
// CHECK-SAME: { ptr @_ZThn8_N1C1gEv, ptr @[[METHOD]],
// CHECK-SAME: { ptr @_ZN1CD2Ev, ptr @[[DTOR]],
// CHECK-SAME: { ptr @_ZN1CD1Ev, ptr @[[DTOR]],
// CHECK-SAME: { ptr @_ZThn8_N1CD1Ev, ptr @[[DTOR]],
// CHECK-SAME: { ptr @_ZN1CD0Ev, ptr @[[DTOR]],
// CHECK-SAME: { ptr @_ZThn8_N1CD0Ev, ptr @[[DTOR]],
