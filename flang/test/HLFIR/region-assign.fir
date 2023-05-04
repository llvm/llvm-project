// Test hlfir.region_assign and hlfir.yield operation parse, verify (no errors),
// and unparse.
// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @region_assign_test(%y : !fir.box<!fir.array<?xi64>>, %x: !fir.box<!fir.array<?xf32>>) {
  %c100 = arith.constant 100 : index
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  hlfir.region_assign {
    %expr = hlfir.elemental %shape : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
    ^bb0(%i : index):
      %yelt = hlfir.designate %y(%i) : (!fir.box<!fir.array<?xi64>>, index) -> !fir.ref<i64>
      %elt = fir.call @some_elemental(%yelt) : (!fir.ref<i64>) -> f32
      hlfir.yield_element %elt : f32
    }
    hlfir.yield %expr : !hlfir.expr<?xf32>
  } to {
    hlfir.yield %x : !fir.box<!fir.array<?xf32>>
  }
  return
}
// CHECK-LABEL:   func.func @region_assign_test(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !fir.box<!fir.array<?xi64>>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: !fir.box<!fir.array<?xf32>>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           hlfir.region_assign {
// CHECK:             %[[VAL_4:.*]] = hlfir.elemental %[[VAL_3]] : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
// CHECK:             ^bb0(%[[VAL_5:.*]]: index):
// CHECK:               %[[VAL_6:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_5]])  : (!fir.box<!fir.array<?xi64>>, index) -> !fir.ref<i64>
// CHECK:               %[[VAL_7:.*]] = fir.call @some_elemental(%[[VAL_6]]) : (!fir.ref<i64>) -> f32
// CHECK:               hlfir.yield_element %[[VAL_7]] : f32
// CHECK:             }
// CHECK:             hlfir.yield %[[VAL_8:.*]] : !hlfir.expr<?xf32>
// CHECK:           } to {
// CHECK:             hlfir.yield %[[VAL_1]] : !fir.box<!fir.array<?xf32>>
// CHECK:           }

func.func @region_user_assign_test(%y : !fir.box<!fir.array<?xi64>>, %x: !fir.box<!fir.array<?xf32>>) {
  hlfir.region_assign {
    hlfir.yield %y : !fir.box<!fir.array<?xi64>>
  } to {
    hlfir.yield %x : !fir.box<!fir.array<?xf32>>
  } user_defined_assign (%rhs : !fir.ref<i64>) to (%lhs : !fir.ref<f32>) {
    %0 = fir.load %rhs : !fir.ref<i64>
    fir.call @user_assign(%lhs, %0) : (!fir.ref<f32>, i64) -> ()
  }
  return
}
// CHECK-LABEL:   func.func @region_user_assign_test(
// CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.box<!fir.array<?xi64>>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: !fir.box<!fir.array<?xf32>>) {
// CHECK:           hlfir.region_assign {
// CHECK:             hlfir.yield %[[VAL_0]] : !fir.box<!fir.array<?xi64>>
// CHECK:           } to {
// CHECK:             hlfir.yield %[[VAL_1]] : !fir.box<!fir.array<?xf32>>
// CHECK:           } user_defined_assign  (%[[VAL_2:.*]]: !fir.ref<i64>) to (%[[VAL_3:.*]]: !fir.ref<f32>) {
// CHECK:             %[[VAL_4:.*]] = fir.load %[[VAL_2]] : !fir.ref<i64>
// CHECK:             fir.call @user_assign(%[[VAL_3]], %[[VAL_4]]) : (!fir.ref<f32>, i64) -> ()
// CHECK:           }

func.func @yield_cleanup(%x: !fir.box<!fir.array<?xf32>>) {
  hlfir.region_assign {
    %0 = fir.allocmem !fir.array<100xf32>
    fir.call @fillin_some_values(%0) : (!fir.heap<!fir.array<100xf32>>) -> ()
    hlfir.yield %0 : !fir.heap<!fir.array<100xf32>> cleanup {
      fir.freemem %0 : !fir.heap<!fir.array<100xf32>>
    }
  } to {
    hlfir.yield %x : !fir.box<!fir.array<?xf32>>
  }
  return
}
// CHECK-LABEL:   func.func @yield_cleanup(
// CHECK-SAME:                             %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>) {
// CHECK:           hlfir.region_assign {
// CHECK:             %[[VAL_1:.*]] = fir.allocmem !fir.array<100xf32>
// CHECK:             fir.call @fillin_some_values(%[[VAL_1]]) : (!fir.heap<!fir.array<100xf32>>) -> ()
// CHECK:             hlfir.yield %[[VAL_1]] : !fir.heap<!fir.array<100xf32>> cleanup {
// CHECK:               fir.freemem %[[VAL_1]] : !fir.heap<!fir.array<100xf32>>
// CHECK:             }
// CHECK:           } to {
// CHECK:             hlfir.yield %[[VAL_0]] : !fir.box<!fir.array<?xf32>>
// CHECK:           }

func.func private @user_assign(!fir.ref<f32>, i64) -> ()
func.func private @some_elemental(!fir.ref<i64>) -> f32
func.func private @fillin_some_values(!fir.heap<!fir.array<100xf32>>) -> ()
