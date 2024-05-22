/// Ensure -fdump-vtable-layout omits the rtti component when passed -fexperimental-omit-vtable-rtti.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-linux-gnu -fno-rtti -fexperimental-omit-vtable-rtti -emit-llvm-only -fdump-vtable-layouts | FileCheck %s

// CHECK:      Vtable for 'A' (2 entries).
// CHECK-NEXT:    0 | offset_to_top (0)
// CHECK-NEXT:        -- (A, 0) vtable address --
// CHECK-NEXT:    1 | void A::foo()

class A {
public:
  virtual void foo();
};

void A::foo() {}

void A_foo(A *a) {
  a->foo();
}
