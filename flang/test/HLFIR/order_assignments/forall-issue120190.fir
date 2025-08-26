// Regression test for https://github.com/llvm/llvm-project/issues/120190
// Verify that hlfir.forall lowering does not try hoisting mask evaluation
// that refer to the forall index inside nested regions only.
// RUN: fir-opt %s --lower-hlfir-ordered-assignments | FileCheck %s

func.func @issue120190(%array: !fir.ref<!fir.array<100xf32>>, %cdt: i1) {
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : i64
  %c50 = arith.constant 50 : i64
  %c100 = arith.constant 100 : i64
  hlfir.forall lb {
    hlfir.yield %c1 : i64
  } ub {
    hlfir.yield %c100 : i64
  }  (%forall_index: i64) {
    hlfir.forall_mask {
      %mask = fir.if %cdt -> i1 {
        // Reference to %forall_index is not directly in
        // hlfir.forall_mask region, but is nested.
        %res = arith.cmpi slt, %forall_index, %c50 : i64
        fir.result %res : i1
      } else {
        %res = arith.cmpi sgt, %forall_index, %c50 : i64
        fir.result %res : i1
      }
      hlfir.yield %mask : i1
    } do {
      hlfir.region_assign {
        hlfir.yield %cst : f32
      } to {
        %6 = hlfir.designate %array (%forall_index)  : (!fir.ref<!fir.array<100xf32>>, i64) -> !fir.ref<f32>
        hlfir.yield %6 : !fir.ref<f32>
      }
    }
  }
  return
}

// CHECK-LABEL: func.func @issue120190(
// CHECK-SAME:                         %[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>,
// CHECK-SAME:                         %[[VAL_1:.*]]: i1) {
// CHECK:         %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[VAL_3:.*]] = arith.constant 1 : i64
// CHECK:         %[[VAL_4:.*]] = arith.constant 50 : i64
// CHECK:         %[[VAL_5:.*]] = arith.constant 100 : i64
// CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
// CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
// CHECK:         %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:         fir.do_loop %[[VAL_9:.*]] = %[[VAL_6]] to %[[VAL_7]] step %[[VAL_8]] {
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (index) -> i64
// CHECK:           %[[VAL_11:.*]] = fir.if %[[VAL_1]] -> (i1) {
// CHECK:             %[[VAL_12:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_4]] : i64
// CHECK:             fir.result %[[VAL_12]] : i1
// CHECK:           } else {
// CHECK:             %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_4]] : i64
// CHECK:             fir.result %[[VAL_13]] : i1
// CHECK:           }
// CHECK:           fir.if %[[VAL_11]] {
// CHECK:             %[[VAL_14:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_10]])  : (!fir.ref<!fir.array<100xf32>>, i64) -> !fir.ref<f32>
// CHECK:             hlfir.assign %[[VAL_2]] to %[[VAL_14]] : f32, !fir.ref<f32>
// CHECK:           }
// CHECK:         }
// CHECK:         return
// CHECK:       }
