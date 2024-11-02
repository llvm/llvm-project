// Cross comdat example
// Both the parent and child VTablea are in their own comdat sections.

// RUN: %clang_cc1 %s -triple=aarch64 -S -o - -emit-llvm -fexperimental-relative-c++-abi-vtables | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64 -S -o - -emit-llvm -fexperimental-relative-c++-abi-vtables | FileCheck %s

// Comdats are emitted for both A and B in this module and for their respective implementations of foo().
// CHECK: $_ZN1A3fooEv = comdat any
// CHECK: $_ZN1B3fooEv = comdat any
// CHECK: $_ZTV1A = comdat any
// CHECK: $_ZTS1A = comdat any
// CHECK: $_ZTI1A = comdat any
// CHECK: $_ZTI1A.rtti_proxy = comdat any
// CHECK: $_ZTV1B = comdat any
// CHECK: $_ZTS1B = comdat any
// CHECK: $_ZTI1B = comdat any
// CHECK: $_ZTI1B.rtti_proxy = comdat any

// Both the vtables for A and B are emitted and in their own comdats.
// CHECK: @_ZTV1A.local = linkonce_odr hidden unnamed_addr constant { [3 x i32] } { [3 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1A.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1A3fooEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, comdat($_ZTV1A), align 4
// CHECK: @_ZTV1B.local = linkonce_odr hidden unnamed_addr constant { [3 x i32] } { [3 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1B.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV1B.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1B3fooEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV1B.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, comdat($_ZTV1B), align 4

// CHECK: @_ZTV1A = linkonce_odr unnamed_addr alias { [3 x i32] }, ptr @_ZTV1A.local
// CHECK: @_ZTV1B = linkonce_odr unnamed_addr alias { [3 x i32] }, ptr @_ZTV1B.local

// CHECK: declare void @_Z5A_fooP1A(ptr noundef)

// The stubs and implementations for foo() are in their own comdat sections.
// CHECK:      define linkonce_odr void @_ZN1A3fooEv(ptr {{.*}}%this) unnamed_addr #{{[0-9]+}} comdat

// CHECK:      define linkonce_odr void @_ZN1B3fooEv(ptr {{.*}}%this) unnamed_addr #{{[0-9]+}} comdat

class A {
public:
  inline virtual void foo() {}
};
class B : public A {
public:
  inline void foo() override {}
};
void A_foo(A *a);

// func() is used so that the vtable for B is accessed when creating the instance.
void func() {
  A a;
  B b;
  A_foo(&a);
  A_foo(&b);
}
