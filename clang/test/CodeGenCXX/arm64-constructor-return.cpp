// RUN: %clang_cc1 %s -triple=arm64-apple-ios7.0.0 -emit-llvm -o - | FileCheck %s

struct S {
  S();
  int iField;
};

S::S() {
  iField = 1;
};

// CHECK: ptr @_ZN1SC2Ev(ptr {{[^,]*}} %this)

// CHECK: ptr @_ZN1SC1Ev(ptr {{[^,]*}} returned align 4 dereferenceable(4) %this)
// CHECK: [[THISADDR:%[a-zA-Z0-9.]+]] = alloca ptr
// CHECK: store ptr %this, ptr [[THISADDR]]
// CHECK: [[THIS1:%.*]] = load ptr, ptr [[THISADDR]]
// CHECK: ret ptr [[THIS1]]
