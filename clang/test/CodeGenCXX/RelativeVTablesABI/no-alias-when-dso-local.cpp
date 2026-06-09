// Check that no alias is emitted when the vtable is already dso_local. This can
// happen if the class is hidden.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -o - -emit-llvm -fhalf-no-semantic-interposition | FileCheck %s --check-prefix=DEFAULT-VIS
// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -o - -emit-llvm -fvisibility=hidden | FileCheck %s --check-prefix=HIDDEN-VIS

// DEFAULT-VIS: @_ZTV1A.local = internal constant
// DEFAULT-VIS: @_ZTV1A ={{.*}}alias { [3 x i32] }, ptr @_ZTV1A.local
// HIDDEN-VIS-NOT: @_ZTV1A.local
// HIDDEN-VIS: @_ZTV1A = hidden constant
class A {
public:
  virtual void func();
};

void A::func() {}
