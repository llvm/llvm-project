// RUN: fir-opt %s | tco | FileCheck %s

// CHECK: @var_x = external global i32
fir.global @var_x : !fir.int<4> {}

// CHECK-LABEL: define i32* @getAddressOfX
func @getAddressOfX() -> !fir.ref<!fir.int<4>> {
  %1 = fir.address_of(@var_x) : !fir.ref<!fir.int<4>>
  // CHECK: ret i32* @var_x
  return %1 : !fir.ref<!fir.int<4>>
}

