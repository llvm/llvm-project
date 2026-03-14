// Test hlfir.get_length operation parse, verify (no errors), and unparse.

// RUN: fir-opt %s | fir-opt | FileCheck %s --check-prefix CHECK --check-prefix CHECK-ALL

// Test canonicalization.
// RUN: fir-opt %s --canonicalize | FileCheck %s --check-prefix CHECK-CANON --check-prefix CHECK-ALL

func.func @fixed_len_scalar(%arg0: !hlfir.expr<!fir.char<1,10>>) -> index {
  %1 = hlfir.get_length %arg0 : (!hlfir.expr<!fir.char<1,10>>) -> index
  return %1 : index
}
// CHECK-ALL-LABEL:   func.func @fixed_len_scalar(
// CHECK-ALL-SAME:      %[[VAL_0:.*]]: !hlfir.expr<!fir.char<1,10>>) -> index {
// CHECK:               %[[VAL_1:.*]] = hlfir.get_length %[[VAL_0]] : (!hlfir.expr<!fir.char<1,10>>) -> index
// CHECK-CANON:         %[[VAL_1:.*]] = arith.constant 10 : index
// CHECK-ALL:           return %[[VAL_1]] : index
// CHECK-ALL:         }

func.func @assumed_len_scalar(%arg0: !hlfir.expr<!fir.char<1,?>>) -> index {
  %1 = hlfir.get_length %arg0 : (!hlfir.expr<!fir.char<1,?>>) -> index
  return %1 : index
}
// CHECK-ALL-LABEL:   func.func @assumed_len_scalar(
// CHECK-ALL-SAME:      %[[VAL_0:.*]]: !hlfir.expr<!fir.char<1,?>>) -> index {
// CHECK-ALL:           %[[VAL_1:.*]] = hlfir.get_length %[[VAL_0]] : (!hlfir.expr<!fir.char<1,?>>) -> index
// CHECK-ALL:           return %[[VAL_1]] : index
// CHECK-ALL:         }

func.func @fixed_len_array(%arg0: !hlfir.expr<?x!fir.char<1,10>>) -> index {
  %1 = hlfir.get_length %arg0 : (!hlfir.expr<?x!fir.char<1,10>>) -> index
  return %1 : index
}
// CHECK-ALL-LABEL:   func.func @fixed_len_array(
// CHECK-ALL-SAME:      %[[VAL_0:.*]]: !hlfir.expr<?x!fir.char<1,10>>) -> index {
// CHECK:               %[[VAL_1:.*]] = hlfir.get_length %[[VAL_0]] : (!hlfir.expr<?x!fir.char<1,10>>) -> index
// CHECK-CANON:         %[[VAL_1:.*]] = arith.constant 10 : index
// CHECK-ALL:           return %[[VAL_1]] : index
// CHECK-ALL:         }

func.func @assumed_len_array(%arg0: !hlfir.expr<?x!fir.char<1,?>>) -> index {
  %1 = hlfir.get_length %arg0 : (!hlfir.expr<?x!fir.char<1,?>>) -> index
  return %1 : index
}
// CHECK-ALL-LABEL:   func.func @assumed_len_array(
// CHECK-ALL-SAME:      %[[VAL_0:.*]]: !hlfir.expr<?x!fir.char<1,?>>) -> index {
// CHECK-ALL:           %[[VAL_1:.*]] = hlfir.get_length %[[VAL_0]] : (!hlfir.expr<?x!fir.char<1,?>>) -> index
// CHECK-ALL:           return %[[VAL_1]] : index
// CHECK-ALL:         }
