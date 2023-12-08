// RUN: tco %s | FileCheck %s
// RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck %s

// CHECK: @var_x = external global i32
fir.global @var_x : !fir.int<4> {}

// CHECK-LABEL: define ptr @getAddressOfX
func.func @getAddressOfX() -> !fir.ref<!fir.int<4>> {
  %1 = fir.address_of(@var_x) : !fir.ref<!fir.int<4>>
  // CHECK: ret ptr @var_x
  return %1 : !fir.ref<!fir.int<4>>
}
