// Check that a vtable is made hidden instead of private if the original vtable
// is not dso_local. The vtable will need to be hidden and not private so it can
// be used as acomdat key signature.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -S -o - -emit-llvm | FileCheck %s

// CHECK: @_ZTV1B.local = linkonce_odr hidden unnamed_addr constant
// CHECK: @_ZTV1B = linkonce_odr unnamed_addr alias { [3 x i32] }, { [3 x i32] }* @_ZTV1B.local

// The VTable will be in a comdat here since it has no key function.
class B {
public:
  inline virtual void func() {}
};

// This is here just to manifest the vtable for B.
void func() {
  B b;
}
