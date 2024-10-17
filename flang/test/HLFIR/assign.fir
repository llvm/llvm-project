// Test hlfir.assign operation parse, verify (no errors), and unparse.

// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @scalar_logical(%arg0: !fir.ref<!fir.logical<1>>, %arg1: !fir.ref<!fir.logical<1>>) {
  hlfir.assign %arg1 to %arg0 : !fir.ref<!fir.logical<1>>, !fir.ref<!fir.logical<1>>
  return
}
// CHECK-LABEL:   func.func @scalar_logical(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.logical<1>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<!fir.logical<1>>) {
// CHECK:  hlfir.assign %[[VAL_1]] to %[[VAL_0]] : !fir.ref<!fir.logical<1>>, !fir.ref<!fir.logical<1>>

func.func @scalar_logical_2(%arg0: !fir.ref<!fir.logical<1>>, %arg1: i1) {
  hlfir.assign %arg1 to %arg0 : i1, !fir.ref<!fir.logical<1>>
  return
}
// CHECK-LABEL:   func.func @scalar_logical_2(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.logical<1>>,
// CHECK-SAME:    %[[VAL_1:.*]]: i1) {
// CHECK:  hlfir.assign %[[VAL_1]] to %[[VAL_0]] : i1, !fir.ref<!fir.logical<1>>

func.func @scalar_integer(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>) {
  hlfir.assign %arg1 to %arg0 : !fir.ref<i32>, !fir.ref<i32>
  return
}
// CHECK-LABEL:   func.func @scalar_integer(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<i32>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<i32>) {
// CHECK:  hlfir.assign %[[VAL_1]] to %[[VAL_0]] : !fir.ref<i32>, !fir.ref<i32>

func.func @scalar_integer_2(%arg0: !fir.ref<i32>, %arg1: i32) {
  hlfir.assign %arg1 to %arg0 : i32, !fir.ref<i32>
  return
}
// CHECK-LABEL:   func.func @scalar_integer_2(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<i32>,
// CHECK-SAME:    %[[VAL_1:.*]]: i32) {
// CHECK:  hlfir.assign %[[VAL_1]] to %[[VAL_0]] : i32, !fir.ref<i32>

func.func @scalar_real(%arg0: !fir.ref<f16>, %arg1: !fir.ref<f16>) {
  hlfir.assign %arg1 to %arg0 : !fir.ref<f16>, !fir.ref<f16>
  return
}
// CHECK-LABEL:   func.func @scalar_real(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<f16>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<f16>) {
// CHECK:  hlfir.assign %[[VAL_1]] to %[[VAL_0]] : !fir.ref<f16>, !fir.ref<f16>

func.func @scalar_complex(%arg0: !fir.ref<complex<f64>>, %arg1: !fir.ref<complex<f64>>) {
  hlfir.assign %arg1 to %arg0 : !fir.ref<complex<f64>>, !fir.ref<complex<f64>>
  return
}
// CHECK-LABEL:   func.func @scalar_complex(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<complex<f64>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<complex<f64>>) {
// CHECK:  hlfir.assign %[[VAL_1]] to %[[VAL_0]] : !fir.ref<complex<f64>>, !fir.ref<complex<f64>>

func.func @scalar_complex_2(%arg0: !fir.ref<complex<f64>>, %arg1: complex<f64>) {
  hlfir.assign %arg1 to %arg0 : complex<f64>, !fir.ref<complex<f64>>
  return
}
// CHECK-LABEL:   func.func @scalar_complex_2(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<complex<f64>>,
// CHECK-SAME:    %[[VAL_1:.*]]: complex<f64>) {
// CHECK:  hlfir.assign %[[VAL_1]] to %[[VAL_0]] : complex<f64>, !fir.ref<complex<f64>>


func.func @scalar_character(%arg0: !fir.boxchar<1>, %arg1: !fir.boxchar<1>) {
  hlfir.assign %arg1 to %arg0 : !fir.boxchar<1>, !fir.boxchar<1>
  return
}
// CHECK-LABEL:   func.func @scalar_character(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<1>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.boxchar<1>) {
// CHECK:  hlfir.assign %[[VAL_1]] to %[[VAL_0]] : !fir.boxchar<1>, !fir.boxchar<1>

func.func @scalar_character_2(%arg0: !fir.ref<!fir.char<1,32>>, %arg1: !fir.ref<!fir.char<1,64>>) {
  hlfir.assign %arg1 to %arg0 : !fir.ref<!fir.char<1,64>>, !fir.ref<!fir.char<1,32>>
  return
}
// CHECK-LABEL:   func.func @scalar_character_2(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.char<1,32>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<!fir.char<1,64>>) {
// CHECK:  hlfir.assign %[[VAL_1]] to %[[VAL_0]] : !fir.ref<!fir.char<1,64>>, !fir.ref<!fir.char<1,32>>

func.func @scalar_character_3(%arg0: !fir.boxchar<1>, %arg1: !hlfir.expr<!fir.char<1,?>>) {
  hlfir.assign %arg1 to %arg0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  return
}
// CHECK-LABEL:   func.func @scalar_character_3(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<1>,
// CHECK-SAME:    %[[VAL_1:.*]]: !hlfir.expr<!fir.char<1,?>>) {
// CHECK:  hlfir.assign %[[VAL_1]] to %[[VAL_0]] : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>

func.func @array(%arg0: !fir.ref<!fir.array<10xi32>>, %arg1: i32) {
  hlfir.assign %arg1 to %arg0 : i32, !fir.ref<!fir.array<10xi32>>
  return
}
// CHECK-LABEL:   func.func @array(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>,
// CHECK-SAME:    %[[VAL_1:.*]]: i32) {
// CHECK:  hlfir.assign %[[VAL_1]] to %[[VAL_0]] : i32, !fir.ref<!fir.array<10xi32>>

func.func @array_2(%arg0: !fir.ref<!fir.array<10xi32>>, %arg1: !hlfir.expr<?xi32>) {
  hlfir.assign %arg1 to %arg0 : !hlfir.expr<?xi32>, !fir.ref<!fir.array<10xi32>>
  return
}
// CHECK-LABEL:   func.func @array_2(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !hlfir.expr<?xi32>) {
// CHECK:  hlfir.assign %[[VAL_1]] to %[[VAL_0]] : !hlfir.expr<?xi32>, !fir.ref<!fir.array<10xi32>>

func.func @array_3(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !hlfir.expr<?xi32>) {
  hlfir.assign %arg1 to %arg0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
  return
}
// CHECK-LABEL:   func.func @array_3(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !hlfir.expr<?xi32>) {
// CHECK:  hlfir.assign %[[VAL_1]] to %[[VAL_0]] : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>

func.func @array_4(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !fir.box<!fir.array<?xi32>>) {
  hlfir.assign %arg1 to %arg0 : !fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>
  return
}
// CHECK-LABEL:   func.func @array_4(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>>) {
// CHECK:  hlfir.assign %[[VAL_1]] to %[[VAL_0]] : !fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>

func.func @alloc_assign(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.box<!fir.array<?xi32>>) {
  hlfir.assign %arg1 to %arg0 realloc : !fir.box<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  return
}
// CHECK-LABEL:   func.func @alloc_assign(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>>) {
// CHECK:  hlfir.assign %[[VAL_1]] to %[[VAL_0]] realloc : !fir.box<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

func.func @alloc_assign_keep_lhs_len(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>, %arg1: !fir.box<!fir.array<?x!fir.char<1,?>>>) {
  hlfir.assign %arg1 to %arg0 realloc keep_lhs_len : !fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
  return
}
// CHECK-LABEL:   func.func @alloc_assign_keep_lhs_len(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>) {
// CHECK:  hlfir.assign %[[VAL_1]] to %[[VAL_0]] realloc keep_lhs_len : !fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
