// RUN: %clang_cc1 -emit-llvm -std=c++11 -o - %s | FileCheck %s

struct D;
struct B {
 virtual D &operator=(D&&) = 0;
};
struct D : B { D(); virtual void a(); };
void D::a() {}
D d;

// CHECK: @_ZTV1D = {{.*}} @_ZN1DaSEOS_ 
// CHECK: define linkonce_odr {{.*}} @_ZN1DaSEOS_
