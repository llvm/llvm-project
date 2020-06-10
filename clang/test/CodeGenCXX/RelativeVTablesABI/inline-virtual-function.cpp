// The VTable is not in a comdat but the inline methods are.
// This doesn’t affect the vtable or the stubs we emit.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O1 -S -o - -emit-llvm -fexperimental-relative-c++-abi-vtables | FileCheck %s

// CHECK: $_ZN1A3fooEv.stub = comdat any
// CHECK: $_ZN1A3barEv.stub = comdat any
// CHECK: $_ZTI1A.rtti_proxy = comdat any

// The vtable has a key function (A::foo()) so it does not have a comdat
// CHECK: @_ZTV1A.local = private unnamed_addr constant { [4 x i32] } { [4 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8* }** @_ZTI1A.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32] }, { [4 x i32] }* @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.A*)* @_ZN1A3fooEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32] }, { [4 x i32] }* @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.A*)* @_ZN1A3barEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32] }, { [4 x i32] }* @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, align 4
// CHECK: @_ZTV1A = unnamed_addr alias { [4 x i32] }, { [4 x i32] }* @_ZTV1A.local

// We do not declare the stub with linkonce_odr so it can be emitted as .globl.
// CHECK:      define hidden void @_ZN1A3barEv.stub(%class.A* %0) unnamed_addr #{{[0-9]+}} comdat {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN1A3barEv(%class.A* %0)
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

class A {
public:
  virtual void foo(); // Key func
  inline virtual void bar() {}
};

void A::foo() {}
