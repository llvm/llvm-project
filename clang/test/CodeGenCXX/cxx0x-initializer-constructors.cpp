// RUN: %clang_cc1 -std=c++11 -S -triple x86_64-none-linux-gnu -emit-llvm -o - %s | FileCheck %s

struct S {
  S(int x) { }
  S(int x, double y, double z) { }
};

void fn1() {
  // CHECK: define void @_Z3fn1v
  S s { 1 };
  // CHECK: alloca %struct.S, align 1
  // CHECK: call void @_ZN1SC1Ei(%struct.S* %s, i32 1)
}

void fn2() {
  // CHECK: define void @_Z3fn2v
  S s { 1, 2.0, 3.0 };
  // CHECK: alloca %struct.S, align 1
  // CHECK: call void @_ZN1SC1Eidd(%struct.S* %s, i32 1, double 2.000000e+00, double 3.000000e+00)
}

void fn3() {
  // CHECK: define void @_Z3fn3v
  S sa[] { { 1 }, { 2 }, { 3 } };
  // CHECK: alloca [3 x %struct.S], align 1
  // CHECK: call void @_ZN1SC1Ei(%struct.S* %{{.+}}, i32 1)
  // CHECK: call void @_ZN1SC1Ei(%struct.S* %{{.+}}, i32 2)
  // CHECK: call void @_ZN1SC1Ei(%struct.S* %{{.+}}, i32 3)
}

void fn4() {
  // CHECK: define void @_Z3fn4v
  S sa[] { { 1, 2.0, 3.0 }, { 4, 5.0, 6.0 } };
  // CHECK: alloca [2 x %struct.S], align 1
  // CHECK: call void @_ZN1SC1Eidd(%struct.S* %{{.+}}, i32 1, double 2.000000e+00, double 3.000000e+00)
  // CHECK: call void @_ZN1SC1Eidd(%struct.S* %{{.+}}, i32 4, double 5.000000e+00, double 6.000000e+00)
}
