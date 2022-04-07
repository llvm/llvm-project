// RUN: %clang_cc1 -no-opaque-pointers %s -triple=arm64-apple-ios7.0.0 -emit-llvm -o - | FileCheck %s
// rdar://12162905

struct S {
  S();
  int iField;
};

S::S() {
  iField = 1;
};

// CHECK: %struct.S* @_ZN1SC2Ev(%struct.S* {{[^,]*}} %this)

// CHECK: %struct.S* @_ZN1SC1Ev(%struct.S* {{[^,]*}} returned align 4 dereferenceable(4) %this)
// CHECK: [[THISADDR:%[a-zA-Z0-9.]+]] = alloca %struct.S*
// CHECK: store %struct.S* %this, %struct.S** [[THISADDR]]
// CHECK: [[THIS1:%.*]] = load %struct.S*, %struct.S** [[THISADDR]]
// CHECK: ret %struct.S* [[THIS1]]
