// RUN: %clang_cc1 %s -triple=i386-apple-darwin10 -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-32
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-64

struct A {
  virtual void f();
  virtual void g();
  virtual void h();
};

void A::f() {}

// CHECK-32: @_ZTV1A = unnamed_addr constant { [5 x i8*] } { [5 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI1A to i8*), i8* bitcast (void (%struct.A*)* @_ZN1A1fEv to i8*), i8* bitcast (void (%struct.A*)* @_ZN1A1gEv to i8*), i8* bitcast (void (%struct.A*)* @_ZN1A1hEv to i8*)] }, align 4
// CHECK-32: @_ZTS1A = constant [3 x i8] c"1A\00", align 1
// CHECK-32: @_ZTI1A = constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i32 2) to i8*), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @_ZTS1A, i32 0, i32 0) }, align 4
// CHECK-64: @_ZTV1A = unnamed_addr constant { [5 x i8*] } { [5 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI1A to i8*), i8* bitcast (void (%struct.A*)* @_ZN1A1fEv to i8*), i8* bitcast (void (%struct.A*)* @_ZN1A1gEv to i8*), i8* bitcast (void (%struct.A*)* @_ZN1A1hEv to i8*)] }, align 8
// CHECK-64: @_ZTS1A = constant [3 x i8] c"1A\00", align 1
// CHECK-64: @_ZTI1A = constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @_ZTS1A, i32 0, i32 0) }, align 8
