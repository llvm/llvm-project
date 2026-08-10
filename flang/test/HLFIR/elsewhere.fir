// Test hlfir.elsewhere operation parse, verify (no errors), and unparse.
// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @test_elsewhere(%mask: !fir.ref<!fir.array<10x!fir.logical<4>>>, %x: !fir.ref<!fir.array<10xf32>>, %y: !fir.box<!fir.array<?xf32>>) {
  hlfir.where {
    hlfir.yield %mask : !fir.ref<!fir.array<10x!fir.logical<4>>>
  } do {
    hlfir.elsewhere do {
      hlfir.region_assign {
        hlfir.yield %y : !fir.box<!fir.array<?xf32>>
      } to {
        hlfir.yield %x : !fir.ref<!fir.array<10xf32>>
      }
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_elsewhere(
// CHECK-SAME:                              %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>,
// CHECK-SAME:                              %[[VAL_1:.*]]: !fir.ref<!fir.array<10xf32>>,
// CHECK-SAME:                              %[[VAL_2:.*]]: !fir.box<!fir.array<?xf32>>) {
// CHECK:           hlfir.where {
// CHECK:             hlfir.yield %[[VAL_0]] : !fir.ref<!fir.array<10x!fir.logical<4>>>
// CHECK:           } do {
// CHECK:             hlfir.elsewhere do {
// CHECK:               hlfir.region_assign {
// CHECK:                 hlfir.yield %[[VAL_2]] : !fir.box<!fir.array<?xf32>>
// CHECK:               } to {
// CHECK:                 hlfir.yield %[[VAL_1]] : !fir.ref<!fir.array<10xf32>>
// CHECK:               }
// CHECK:             }
// CHECK:           }

func.func @test_masked_elsewhere(%mask: !fir.ref<!fir.array<10x!fir.logical<4>>>, %x: !fir.ref<!fir.array<10xf32>>, %y: !fir.box<!fir.array<?xf32>>) {
  hlfir.where {
    hlfir.yield %mask : !fir.ref<!fir.array<10x!fir.logical<4>>>
  } do {
    hlfir.elsewhere mask {
      %other_mask = fir.call @get_mask() : () -> !fir.ptr<!fir.array<10x!fir.logical<4>>>
      hlfir.yield %other_mask : !fir.ptr<!fir.array<10x!fir.logical<4>>>
    } do {
      hlfir.region_assign {
        hlfir.yield %y : !fir.box<!fir.array<?xf32>>
      } to {
        hlfir.yield %x : !fir.ref<!fir.array<10xf32>>
      }
      hlfir.elsewhere do {
        hlfir.region_assign {
          hlfir.yield %x : !fir.ref<!fir.array<10xf32>>
        } to {
          hlfir.yield %y : !fir.box<!fir.array<?xf32>>
        }
      }
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_masked_elsewhere(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !fir.ref<!fir.array<10xf32>>,
// CHECK-SAME:                                     %[[VAL_2:.*]]: !fir.box<!fir.array<?xf32>>) {
// CHECK:           hlfir.where {
// CHECK:             hlfir.yield %[[VAL_0]] : !fir.ref<!fir.array<10x!fir.logical<4>>>
// CHECK:           } do {
// CHECK:             hlfir.elsewhere mask {
// CHECK:               %[[VAL_3:.*]] = fir.call @get_mask() : () -> !fir.ptr<!fir.array<10x!fir.logical<4>>>
// CHECK:               hlfir.yield %[[VAL_3]] : !fir.ptr<!fir.array<10x!fir.logical<4>>>
// CHECK:             } do {
// CHECK:               hlfir.region_assign {
// CHECK:                 hlfir.yield %[[VAL_2]] : !fir.box<!fir.array<?xf32>>
// CHECK:               } to {
// CHECK:                 hlfir.yield %[[VAL_1]] : !fir.ref<!fir.array<10xf32>>
// CHECK:               }
// CHECK:               hlfir.elsewhere do {
// CHECK:                 hlfir.region_assign {
// CHECK:                   hlfir.yield %[[VAL_1]] : !fir.ref<!fir.array<10xf32>>
// CHECK:                 } to {
// CHECK:                   hlfir.yield %[[VAL_2]] : !fir.box<!fir.array<?xf32>>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
