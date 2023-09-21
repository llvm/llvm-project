// Check the layout of the vtable for a normal class.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O1 -S -o - -emit-llvm -fhalf-no-semantic-interposition | FileCheck %s

// We should be emitting comdats for each of the virtual function RTTI proxies
// CHECK: $_ZTI1A.rtti_proxy = comdat any

// VTable contains offsets and references to the hidden symbols
// The vtable definition itself is private so we can take relative references to
// it. The vtable symbol will be exposed through a public alias.
// CHECK: @_ZTV1A.local = private unnamed_addr constant { [3 x i32] } { [3 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1A.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1A3fooEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, align 4
// CHECK: @_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
// CHECK: @_ZTS1A ={{.*}} constant [3 x i8] c"1A\00", align 1
// CHECK: @_ZTI1A ={{.*}} constant { ptr, ptr } { ptr getelementptr inbounds (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i32 8), ptr @_ZTS1A }, align 8

// The rtti should be in a comdat
// CHECK: @_ZTI1A.rtti_proxy = hidden unnamed_addr constant ptr @_ZTI1A, comdat

// The vtable symbol is exposed through an alias.
// @_ZTV1A = dso_local unnamed_addr alias { [3 x i32] }, ptr @_ZTV1A.local

class A {
public:
  virtual void foo();
};

void A::foo() {}

void A_foo(A *a) {
  a->foo();
}
