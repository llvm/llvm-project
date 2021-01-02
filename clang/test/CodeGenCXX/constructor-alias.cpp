// RUN: %clang_cc1 -emit-llvm -triple mipsel--linux-gnu -mconstructor-aliases -o - %s | FileCheck %s

// The target attribute code used to get confused with aliases. Make sure
// we don't crash when an alias is used.

struct B {
  B();
};
B::B() {
}

// CHECK: @_ZN1BC1Ev ={{.*}} unnamed_addr alias void (%struct.B*), void (%struct.B*)* @_ZN1BC2Ev
