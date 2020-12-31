// Check the layout of the vtable for a normal class.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O1 -S -o - -emit-llvm -fexperimental-relative-c++-abi-vtables -fhalf-no-semantic-interposition | FileCheck %s

// We should be emitting comdats for each of the virtual function RTTI proxies
// CHECK: $_ZTI1A.rtti_proxy = comdat any

// VTable contains offsets and references to the hidden symbols
// The vtable definition itself is private so we can take relative references to
// it. The vtable symbol will be exposed through a public alias.
// CHECK: @_ZTV1A.local = private unnamed_addr constant { [3 x i32] } { [3 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8* }** @_ZTI1A.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [3 x i32] }, { [3 x i32] }* @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.A*)* dso_local_equivalent @_ZN1A3fooEv to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [3 x i32] }, { [3 x i32] }* @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, align 4
// CHECK: @_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
// CHECK: @_ZTS1A ={{.*}} constant [3 x i8] c"1A\00", align 1
// CHECK: @_ZTI1A ={{.*}} constant { i8*, i8* } { i8* getelementptr inbounds (i8, i8* bitcast (i8** @_ZTVN10__cxxabiv117__class_type_infoE to i8*), i32 8), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @_ZTS1A, i32 0, i32 0) }, align 8

// The rtti should be in a comdat
// CHECK: @_ZTI1A.rtti_proxy = hidden unnamed_addr constant { i8*, i8* }* @_ZTI1A, comdat

// The vtable symbol is exposed through an alias.
// @_ZTV1A = dso_local unnamed_addr alias { [3 x i32] }, { [3 x i32] }* @_ZTV1A.local

class A {
public:
  virtual void foo();
};

void A::foo() {}

void A_foo(A *a) {
  a->foo();
}
