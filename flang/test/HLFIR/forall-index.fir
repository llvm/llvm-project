// Test hlfir.forall_index operation parse, verify (no errors), unparse,
// and canonicalization.
// RUN: fir-opt %s | fir-opt | FileCheck %s
// RUN: fir-opt -canonicalize %s | FileCheck %s --check-prefix=CANONICALIZATION

func.func @forall_index(%x: !fir.ref<!fir.array<10xf32>>, %y: !fir.ref<!fir.array<10xf32>>) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  hlfir.forall lb {
    hlfir.yield %c1 : index
  } ub {
    hlfir.yield %c10 : index
  } (%arg0: i64) {
    %i = hlfir.forall_index "i" %arg0 : (i64) -> !fir.ref<i64>
    hlfir.region_assign {
      %ival = fir.load %i : !fir.ref<i64>
      %yi = hlfir.designate %y(%ival) : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
      hlfir.yield %yi : !fir.ref<f32>
    } to {
      %ival = fir.load %i : !fir.ref<i64>
      %xi = hlfir.designate %x(%ival) : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
      hlfir.yield %xi : !fir.ref<f32>
    }
  }
  return
}
// CHECK-LABEL:  func.func @forall_index(
// CHECK:           hlfir.forall lb {
// CHECK:           } ub {
// CHECK:           }  (%[[VAL_4:.*]]: i64) {
// CHECK:             hlfir.forall_index "i" %[[VAL_4]] : (i64) -> !fir.ref<i64>

// CANONICALIZATION-LABEL:  func.func @forall_index(
// CANONICALIZATION:      hlfir.forall lb {
// CANONICALIZATION:      } ub {
// CANONICALIZATION:      }  (%[[VAL_4:.*]]: i64) {
// CANONICALIZATION-NOT:    hlfir.forall_index
// CANONICALIZATION:        hlfir.designate %{{.*}} (%[[VAL_4]])  : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
// CANONICALIZATION:        hlfir.designate %{{.*}} (%[[VAL_4]])  : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>

func.func @forall_index_do_not_canonicalize(%x: !fir.ref<!fir.array<10xf32>>, %y: !fir.ref<!fir.array<10xf32>>) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  hlfir.forall lb {
    hlfir.yield %c1 : index
  } ub {
    hlfir.yield %c10 : index
  } (%arg0: i64) {
    %i = hlfir.forall_index "i" %arg0 : (i64) -> !fir.ref<i64>
    hlfir.region_assign {
      %res = fir.call @taking_address(%i) : (!fir.ref<i64>) -> f32
      hlfir.yield %res : f32
    } to {
      %ival = fir.load %i : !fir.ref<i64>
      %xi = hlfir.designate %x(%ival) : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
      hlfir.yield %xi : !fir.ref<f32>
    }
  }
  return
}
// CHECK-LABEL:  func.func @forall_index_do_not_canonicalize(
// CHECK:           hlfir.forall lb {
// CHECK:           } ub {
// CHECK:           }  (%[[VAL_4:.*]]: i64) {
// CHECK:             hlfir.forall_index "i" %[[VAL_4]] : (i64) -> !fir.ref<i64>

// CANONICALIZATION-LABEL:  func.func @forall_index_do_not_canonicalize(
// CANONICALIZATION:  %[[VAL_5:.*]] = hlfir.forall_index "i" %[[VAL_4:.*]] : (i64) -> !fir.ref<i64>
// CANONICALIZATION:  fir.call @taking_address(%[[VAL_5]]) : (!fir.ref<i64>) -> f32
// CANONICALIZATION:  %[[VAL_6:.*]] = fir.load %[[VAL_5]] : !fir.ref<i64>
// CANONICALIZATION:  hlfir.designate %{{.*}} (%[[VAL_6]])  : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
