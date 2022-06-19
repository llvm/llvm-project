// Check that no alias is emitted when the vtable is already dso_local. This can
// happen if the class is hidden.

// RUN: %clang_cc1 -no-opaque-pointers %s -triple=aarch64-unknown-fuchsia -S -o - -emit-llvm -fhalf-no-semantic-interposition | FileCheck %s --check-prefix=DEFAULT-VIS
// RUN: %clang_cc1 -no-opaque-pointers %s -triple=aarch64-unknown-fuchsia -S -o - -emit-llvm -fvisibility hidden | FileCheck %s --check-prefix=HIDDEN-VIS

// DEFAULT-VIS: @_ZTV1A.local = private unnamed_addr constant
// DEFAULT-VIS: @_ZTV1A ={{.*}} unnamed_addr alias { [3 x i32] }, { [3 x i32] }* @_ZTV1A.local
// HIDDEN-VIS-NOT: @_ZTV1A.local
// HIDDEN-VIS: @_ZTV1A = hidden unnamed_addr constant
class A {
public:
  virtual void func();
};

void A::func() {}
