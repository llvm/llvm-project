// Inline comdat method definition example.
// The VTable is in a comdat and defined anywhere the inline definition is.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O1 -S -o - -emit-llvm -fexperimental-relative-c++-abi-vtables | FileCheck %s

// CHECK: $_ZN1A3fooEv.stub = comdat any
// CHECK: $_ZTV1A = comdat any
// CHECK: $_ZTS1A = comdat any
// CHECK: $_ZTI1A = comdat any
// CHECK: $_ZTI1A.rtti_proxy = comdat any

// The VTable is linkonce_odr and in a comdat here bc it’s key function is inline defined.
// CHECK: @_ZTV1A.local = linkonce_odr hidden unnamed_addr constant { [3 x i32] } { [3 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8* }** @_ZTI1A.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [3 x i32] }, { [3 x i32] }* @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.A*)* @_ZN1A3fooEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [3 x i32] }, { [3 x i32] }* @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, comdat($_ZTV1A), align 4
// CHECK: @_ZTV1A = linkonce_odr unnamed_addr alias { [3 x i32] }, { [3 x i32] }* @_ZTV1A.local

// CHECK:      define hidden void @_ZN1A3fooEv.stub(%class.A* %0) unnamed_addr #{{[0-9]+}} comdat
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN1A3fooEv(%class.A* %0)
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

class A {
public:
  virtual void foo();
};
void A_foo(A *a);

inline void A::foo() {}

void func() {
  A a;
  A_foo(&a);
}
