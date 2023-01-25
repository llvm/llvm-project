// RUN: %clang_cc1 %s -triple=i386-apple-darwin10 -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-32
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-64

struct A {
  virtual void f();
  virtual void g();
  virtual void h();
};

void A::f() {}

// CHECK-32: @_ZTV1A ={{.*}} unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTI1A, ptr @_ZN1A1fEv, ptr @_ZN1A1gEv, ptr @_ZN1A1hEv] }, align 4
// CHECK-32: @_ZTS1A ={{.*}} constant [3 x i8] c"1A\00", align 1
// CHECK-32: @_ZTI1A ={{.*}} constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i32 2), ptr @_ZTS1A }, align 4
// CHECK-64: @_ZTV1A ={{.*}} unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTI1A, ptr @_ZN1A1fEv, ptr @_ZN1A1gEv, ptr @_ZN1A1hEv] }, align 8
// CHECK-64: @_ZTS1A ={{.*}} constant [3 x i8] c"1A\00", align 1
// CHECK-64: @_ZTI1A ={{.*}} constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1A }, align 8
