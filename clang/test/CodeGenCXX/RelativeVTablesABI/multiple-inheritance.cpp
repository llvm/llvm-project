// Multiple inheritance.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O1 -S -o - -emit-llvm -fexperimental-relative-c++-abi-vtables | FileCheck %s

// CHECK: %class.C = type { %class.A, %class.B }
// CHECK: %class.A = type { i32 (...)** }
// CHECK: %class.B = type { i32 (...)** }

// VTable for C contains 2 sub-vtables (represented as 2 structs). The first contains the components for B and the second contains the components for C. The RTTI ptr in both arrays still point to the RTTI struct for C.
// The component for bar() instead points to a thunk which redirects to C::bar() which overrides B::bar().
// Now that we have a class with 2 parents, the offset to top in the second array is non-zero.
// CHECK: @_ZTV1C.local = private unnamed_addr constant { [4 x i32], [3 x i32] } { [4 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8*, i32, i32, i8*, i64, i8*, i64 }** @_ZTI1C.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32], [3 x i32] }, { [4 x i32], [3 x i32] }* @_ZTV1C.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.C*)* @_ZN1C3fooEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32], [3 x i32] }, { [4 x i32], [3 x i32] }* @_ZTV1C.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.C*)* @_ZN1C3barEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32], [3 x i32] }, { [4 x i32], [3 x i32] }* @_ZTV1C.local, i32 0, i32 0, i32 2) to i64)) to i32)], [3 x i32] [i32 -8, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8*, i32, i32, i8*, i64, i8*, i64 }** @_ZTI1C.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32], [3 x i32] }, { [4 x i32], [3 x i32] }* @_ZTV1C.local, i32 0, i32 1, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.C*)* @_ZThn8_N1C3barEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32], [3 x i32] }, { [4 x i32], [3 x i32] }* @_ZTV1C.local, i32 0, i32 1, i32 2) to i64)) to i32)] }, align 4

// CHECK: @_ZTV1C = unnamed_addr alias { [4 x i32], [3 x i32] }, { [4 x i32], [3 x i32] }* @_ZTV1C.local

// CHECK:      define void @_Z8C_foobarP1C(%class.C* %c) local_unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[c:%[0-9]+]] = bitcast %class.C* %c to i8**
// CHECK-NEXT:   [[vtable:%[a-z0-9]+]] = load i8*, i8** [[c]], align 8

// Offset 0 to get first method
// CHECK-NEXT:   [[ptr1:%[0-9]+]] = call i8* @llvm.load.relative.i32(i8* [[vtable]], i32 0)
// CHECK-NEXT:   [[method1:%[0-9]+]] = bitcast i8* [[ptr1]] to void (%class.C*)*
// CHECK-NEXT:   call void [[method1]](%class.C* {{[^,]*}} %c)
// CHECK-NEXT:   [[vtable:%[a-z0-9]+]] = load i8*, i8** [[c]], align 8

// Offset by 4 to get the next bar()
// CHECK-NEXT:   [[ptr2:%[0-9]+]] = call i8* @llvm.load.relative.i32(i8* [[vtable]], i32 4)
// CHECK-NEXT:   [[method2:%[0-9]+]] = bitcast i8* [[ptr2]] to void (%class.C*)*
// CHECK-NEXT:   call void [[method2]](%class.C* {{[^,]*}} %c)
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

class A {
public:
  virtual void foo();
};

class B {
  virtual void bar();
};

class C : public A, public B {
public:
  void foo() override;
  void bar() override;
};

void C::foo() {}
void C::bar() {}

void C_foobar(C *c) {
  c->foo();
  c->bar();
}
