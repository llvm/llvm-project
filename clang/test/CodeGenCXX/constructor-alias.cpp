// RUN: %clang_cc1 -emit-llvm -triple mipsel--linux-gnu -mno-constructor-aliases -mconstructor-aliases -o - %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -triple mipsel--linux-gnu -mconstructor-aliases -mno-constructor-aliases -o - %s | FileCheck %s --check-prefix=NO-ALIAS

// The target attribute code used to get confused with aliases. Make sure
// we don't crash when an alias is used.

struct B {
  B();
};
B::B() {
}

// CHECK: @_ZN1BC1Ev ={{.*}} unnamed_addr alias void (%struct.B*), void (%struct.B*)* @_ZN1BC2Ev
// NO-ALIAS-NOT: @_ZN1BC1Ev ={{.*}} unnamed_addr alias void (%struct.B*), void (%struct.B*)* @_ZN1BC2Ev
