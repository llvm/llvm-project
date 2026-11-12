// Test hlfir.forall operation parse, verify (no errors), and unparse.
// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @forall_test(%x: !fir.box<!fir.array<?xf32>>, %x2: !fir.box<!fir.array<?x?xf32>>) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  hlfir.forall lb {
    hlfir.yield %c1 : index
  } ub {
    hlfir.yield %c10 : index
  } (%i : index) {
    hlfir.region_assign {
      %res = fir.call @foo(%i) : (index) -> f32
      hlfir.yield %res : f32
    } to {
      %xi = hlfir.designate %x(%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
      hlfir.yield %xi : !fir.ref<f32>
    }
    hlfir.forall lb {
      hlfir.yield %c1 : index
    } ub {
      hlfir.yield %c10 : index
    } (%j : index) {
      hlfir.region_assign {
        %jf = fir.convert %j : (index) -> f32
        hlfir.yield %jf : f32
      } to {
        %xij = hlfir.designate %x2(%i, %j) : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
        hlfir.yield %xij : !fir.ref<f32>
      }
    }
  }
  return
}
// CHECK-LABEL:   func.func @forall_test(
// CHECK-SAME:                           %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>,
// CHECK-SAME:                           %[[VAL_1:.*]]: !fir.box<!fir.array<?x?xf32>>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
// CHECK:           hlfir.forall lb {
// CHECK:             hlfir.yield %[[VAL_2]] : index
// CHECK:           } ub {
// CHECK:             hlfir.yield %[[VAL_3]] : index
// CHECK:           }  (%[[VAL_4:.*]]: index) {
// CHECK:             hlfir.region_assign {
// CHECK:               %[[VAL_5:.*]] = fir.call @foo(%[[VAL_4]]) : (index) -> f32
// CHECK:               hlfir.yield %[[VAL_5]] : f32
// CHECK:             } to {
// CHECK:               %[[VAL_6:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_4]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:               hlfir.yield %[[VAL_6]] : !fir.ref<f32>
// CHECK:             }
// CHECK:             hlfir.forall lb {
// CHECK:               hlfir.yield %[[VAL_2]] : index
// CHECK:             } ub {
// CHECK:               hlfir.yield %[[VAL_3]] : index
// CHECK:             }  (%[[VAL_7:.*]]: index) {
// CHECK:               hlfir.region_assign {
// CHECK:                 %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (index) -> f32
// CHECK:                 hlfir.yield %[[VAL_8]] : f32
// CHECK:               } to {
// CHECK:                 %[[VAL_9:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_4]], %[[VAL_7]])  : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
// CHECK:                 hlfir.yield %[[VAL_9]] : !fir.ref<f32>
// CHECK:               }
// CHECK:             }
// CHECK:           }

func.func @forall_test_step(%x : !fir.box<!fir.array<10xf32>>, %y: !fir.box<!fir.array<?xf32>>) {
  hlfir.forall lb {
    %c1 = arith.constant 1 : index
    hlfir.yield %c1 : index
  } ub {
    %c10 = arith.constant 10 : index
    hlfir.yield %c10 : index
  } step {
    %c2 = arith.constant 2 : index
    hlfir.yield %c2 : index
  } (%i : index) {
    hlfir.region_assign {
      %yi = hlfir.designate %y(%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
      %val = fir.load %yi : !fir.ref<f32>
      hlfir.yield %val : f32
    } to {
      %xi = hlfir.designate %x(%i) : (!fir.box<!fir.array<10xf32>>, index) -> !fir.ref<f32>
      hlfir.yield %xi : !fir.ref<f32>
    }
  }
  return
}
// CHECK-LABEL:   func.func @forall_test_step(
// CHECK-SAME:                                %[[VAL_0:.*]]: !fir.box<!fir.array<10xf32>>,
// CHECK-SAME:                                %[[VAL_1:.*]]: !fir.box<!fir.array<?xf32>>) {
// CHECK:           hlfir.forall lb {
// CHECK:             %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:             hlfir.yield %[[VAL_2]] : index
// CHECK:           } ub {
// CHECK:             %[[VAL_3:.*]] = arith.constant 10 : index
// CHECK:             hlfir.yield %[[VAL_3]] : index
// CHECK:           } step {
// CHECK:             %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK:             hlfir.yield %[[VAL_4]] : index
// CHECK:           }  (%[[VAL_5:.*]]: index) {
// CHECK:             hlfir.region_assign {
// CHECK:               %[[VAL_6:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_5]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_7:.*]] = fir.load %[[VAL_6]] : !fir.ref<f32>
// CHECK:               hlfir.yield %[[VAL_7]] : f32
// CHECK:             } to {
// CHECK:               %[[VAL_8:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_5]])  : (!fir.box<!fir.array<10xf32>>, index) -> !fir.ref<f32>
// CHECK:               hlfir.yield %[[VAL_8]] : !fir.ref<f32>
// CHECK:             }
// CHECK:           }
