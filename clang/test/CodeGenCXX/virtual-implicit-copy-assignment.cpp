// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

struct D;
struct B {
 virtual D& operator = (const D&);
};
struct D : B { D(); virtual void a(); };
void D::a() {}

// CHECK: @_ZTV1D = {{.*}} @_ZN1DaSERKS_ 
// CHECK: define linkonce_odr {{.*}} @_ZN1DaSERKS_
