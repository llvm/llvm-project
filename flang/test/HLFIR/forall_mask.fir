// Test hlfir.forall_mask operation parse, verify (no errors), and unparse.
// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @forall_mask_test(%x: !fir.box<!fir.array<?xf32>>) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  hlfir.forall lb {
    hlfir.yield %c1 : index
  } ub {
    hlfir.yield %c10 : index
  } (%i : index) {
    hlfir.forall_mask {
      %mask = fir.call @some_condition(%i) : (index) -> i1
      hlfir.yield %mask : i1
    } do {
      hlfir.region_assign {
        %res = fir.call @foo(%i) : (index) -> f32
        hlfir.yield %res : f32
      } to {
        %xi = hlfir.designate %x(%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
        hlfir.yield %xi : !fir.ref<f32>
      }
    }
  }
  return
}

// CHECK-LABEL:   func.func @forall_mask_test(
// CHECK-SAME:                                %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:           hlfir.forall lb {
// CHECK:             hlfir.yield %[[VAL_1]] : index
// CHECK:           } ub {
// CHECK:             hlfir.yield %[[VAL_2]] : index
// CHECK:           }  (%[[VAL_3:.*]]: index) {
// CHECK:             hlfir.forall_mask {
// CHECK:               %[[VAL_4:.*]] = fir.call @some_condition(%[[VAL_3]]) : (index) -> i1
// CHECK:               hlfir.yield %[[VAL_4]] : i1
// CHECK:             } do {
// CHECK:               hlfir.region_assign {
// CHECK:                 %[[VAL_5:.*]] = fir.call @foo(%[[VAL_3]]) : (index) -> f32
// CHECK:                 hlfir.yield %[[VAL_5]] : f32
// CHECK:               } to {
// CHECK:                 %[[VAL_6:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_3]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:                 hlfir.yield %[[VAL_6]] : !fir.ref<f32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
