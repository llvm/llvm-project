// Test hlfir.copy_in and hlfir.copy_out operation parse, verify (no errors),
// and unparse.

// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @test_copy_in(%box: !fir.box<!fir.array<?xf64>>, %temp: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, %is_present: i1) {
  %0:2 = hlfir.copy_in %box to %temp : (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> (!fir.box<!fir.array<?xf64>>, i1)
  %1:2 = hlfir.copy_in %box to %temp handle_optional %is_present : (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, i1) -> (!fir.box<!fir.array<?xf64>>, i1)
  return
}
// CHECK-LABEL:   func.func @test_copy_in(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf64>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>,
// CHECK-SAME:    %[[VAL_2:.*]]: i1
// CHECK:  hlfir.copy_in %[[VAL_0]] to %[[VAL_1]] : (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> (!fir.box<!fir.array<?xf64>>, i1)
// CHECK:  hlfir.copy_in %[[VAL_0]] to %[[VAL_1]] handle_optional %[[VAL_2]] : (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, i1) -> (!fir.box<!fir.array<?xf64>>, i1)

func.func @test_copy_out(%box: !fir.box<!fir.array<?xf64>>, %temp: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, %was_copied: i1) {
  hlfir.copy_out %temp, %was_copied : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, i1) -> ()
  hlfir.copy_out %temp, %was_copied to %box : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, i1, !fir.box<!fir.array<?xf64>>) -> ()
  return
}
// CHECK-LABEL:   func.func @test_copy_out(
// CHECK-SAME:    %[[VAL_0:[^:]*]]: !fir.box<!fir.array<?xf64>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>,
// CHECK-SAME:    %[[VAL_2:.*]]: i1) {
// CHECK:  hlfir.copy_out %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, i1) -> ()
// CHECK:  hlfir.copy_out %[[VAL_1]], %[[VAL_2]] to %[[VAL_0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, i1, !fir.box<!fir.array<?xf64>>) -> ()
