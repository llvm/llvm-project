// Test hlfir.where operation parse, verify (no errors), and unparse.
// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @test_where(%mask: !fir.ref<!fir.array<10x!fir.logical<4>>>, %x: !fir.ref<!fir.array<10xf32>>, %y: !fir.box<!fir.array<?xf32>>) {
  hlfir.where {
    hlfir.yield %mask : !fir.ref<!fir.array<10x!fir.logical<4>>>
  } do {
    hlfir.region_assign {
      hlfir.yield %y : !fir.box<!fir.array<?xf32>>
    } to {
      hlfir.yield %x : !fir.ref<!fir.array<10xf32>>
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_where(
// CHECK-SAME:                          %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>,
// CHECK-SAME:                          %[[VAL_1:.*]]: !fir.ref<!fir.array<10xf32>>,
// CHECK-SAME:                          %[[VAL_2:.*]]: !fir.box<!fir.array<?xf32>>) {
// CHECK:           hlfir.where {
// CHECK:             hlfir.yield %[[VAL_0]] : !fir.ref<!fir.array<10x!fir.logical<4>>>
// CHECK:           } do {
// CHECK:             hlfir.region_assign {
// CHECK:               hlfir.yield %[[VAL_2]] : !fir.box<!fir.array<?xf32>>
// CHECK:             } to {
// CHECK:               hlfir.yield %[[VAL_1]] : !fir.ref<!fir.array<10xf32>>
// CHECK:             }
// CHECK:           }
