// Cross comdat example
// Child VTable is in a comdat section.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O1 -S -o - -emit-llvm | FileCheck %s

// A comdat is emitted for B but not A
// CHECK-DAG: $_ZTV1B = comdat any
// CHECK-DAG: $_ZTS1B = comdat any
// CHECK-DAG: $_ZTI1B = comdat any
// CHECK-DAG: $_ZTI1B.rtti_proxy = comdat any
// CHECK-DAG: $_ZTI1A.rtti_proxy = comdat any

// VTable for B is emitted here since we access it when creating an instance of B. The VTable is also linkonce_odr and in its own comdat.
// CHECK-DAG: @_ZTV1B.local = linkonce_odr hidden unnamed_addr constant { [3 x i32] } { [3 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1B.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV1B.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1B3fooEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV1B.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, comdat($_ZTV1B), align 4

// The RTTI objects aren’t that important, but it is good to know that they are emitted here since they are used in the vtable for B, and external references are used for RTTI stuff from A.
// CHECK-DAG: @_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
// CHECK-DAG: @_ZTS1B =
// CHECK-DAG: @_ZTI1A =
// CHECK-DAG: @_ZTI1B =
// CHECK-DAG: @_ZTI1B.rtti_proxy = hidden unnamed_addr constant ptr @_ZTI1B, comdat

// We will emit a vtable for B here, so it does have an alias, but we will not
// emit one for A.
// CHECK: @_ZTV1B = linkonce_odr unnamed_addr alias { [3 x i32] }, ptr @_ZTV1B.local
// CHECK-NOT: @_ZTV1A = {{.*}}alias

class A {
public:
  virtual void foo();
};
class B : public A {
public:
  inline void foo() override {}
};
void A_foo(A *a);

// func() is used so that the vtable for B is accessed when creating the instance.
void func() {
  B b;
  A_foo(&b);
}
