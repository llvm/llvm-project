// RUN: %clang_cc1 -fno-rtti -emit-llvm -triple=i386-pc-win32 -fms-extensions -fms-compatibility -std=c++11 %s -o - | FileCheck %s

namespace PR23452 {
struct A {
    virtual void f();
};
struct B : virtual A {
    virtual void f();
};
void (B::*MemPtr)(void) = &B::f;
// CHECK-DAG: @"?MemPtr@PR23452@@3P8B@1@AEXXZQ21@" = dso_local global { i8*, i32, i32 } { i8* bitcast ({{.*}} @"??_9B@PR23452@@$BA@AE" to i8*), i32 0, i32 4 }
}
