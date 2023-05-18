// Test hlfir.associate and hlfir.end_associate operations parse, verify
// (no errors), and unparse.

// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @test_cst_char(%arg0: !hlfir.expr<!fir.char<1,12>>) {
  %c12 = arith.constant 12 : index
  %0:3 = hlfir.associate %arg0 typeparams %c12 {uniq_name = "x"} : (!hlfir.expr<!fir.char<1,12>>, index) -> (!fir.ref<!fir.char<1,12>>, !fir.ref<!fir.char<1,12>>, i1)
  fir.call @foo(%0#0) : (!fir.ref<!fir.char<1,12>>) -> ()
  hlfir.end_associate %0#1, %0#2 : !fir.ref<!fir.char<1,12>>, i1
  return
}
func.func private @foo(!fir.ref<!fir.char<1,12>>)
// CHECK-LABEL:   func.func @test_cst_char(
// CHECK-SAME:    %[[VAL_0:.*]]: !hlfir.expr<!fir.char<1,12>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 12 : index
// CHECK:  %[[VAL_2:.*]]:3 = hlfir.associate %[[VAL_0]] typeparams %[[VAL_1]] {uniq_name = "x"} : (!hlfir.expr<!fir.char<1,12>>, index) -> (!fir.ref<!fir.char<1,12>>, !fir.ref<!fir.char<1,12>>, i1)
// CHECK:  fir.call @foo(%[[VAL_2]]#0) : (!fir.ref<!fir.char<1,12>>) -> ()
// CHECK:  hlfir.end_associate %[[VAL_2]]#1, %[[VAL_2]]#2 : !fir.ref<!fir.char<1,12>>, i1


func.func @test_dyn_char(%arg0: !hlfir.expr<!fir.char<1,?>>) {
  %c12 = arith.constant 12 : index
  %0:3 = hlfir.associate %arg0 typeparams %c12 {uniq_name = "x"} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  fir.call @foo2(%0#0) : (!fir.boxchar<1>) -> ()
  hlfir.end_associate %0#1, %0#2 : !fir.ref<!fir.char<1,?>>, i1
  return
}
func.func private @foo2(!fir.boxchar<1>)
// CHECK-LABEL:   func.func @test_dyn_char(
// CHECK-SAME:    %[[VAL_0:.*]]: !hlfir.expr<!fir.char<1,?>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 12 : index
// CHECK:  %[[VAL_2:.*]]:3 = hlfir.associate %[[VAL_0]] typeparams %[[VAL_1]] {uniq_name = "x"} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
// CHECK:  fir.call @foo2(%[[VAL_2]]#0) : (!fir.boxchar<1>) -> ()
// CHECK:  hlfir.end_associate %[[VAL_2]]#1, %[[VAL_2]]#2 : !fir.ref<!fir.char<1,?>>, i1


func.func @test_integer(%arg0: i32) {
  %0:3 = hlfir.associate %arg0 {uniq_name = "x"} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
  fir.call @foo3(%0#0) : (!fir.ref<i32>) -> ()
  hlfir.end_associate %0#1, %0#2 : !fir.ref<i32>, i1
  return
}
func.func private @foo3(!fir.ref<i32>)
// CHECK-LABEL:   func.func @test_integer(
// CHECK-SAME:    %[[VAL_0:.*]]: i32) {
// CHECK:  %[[VAL_1:.*]]:3 = hlfir.associate %[[VAL_0]] {uniq_name = "x"} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
// CHECK:  fir.call @foo3(%[[VAL_1]]#0) : (!fir.ref<i32>) -> ()
// CHECK:  hlfir.end_associate %[[VAL_1]]#1, %[[VAL_1]]#2 : !fir.ref<i32>, i1


func.func @test_logical(%arg0: !fir.logical<8>) {
  %0:3 = hlfir.associate %arg0 {uniq_name = "x"} : (!fir.logical<8>) -> (!fir.ref<!fir.logical<8>>, !fir.ref<!fir.logical<8>>, i1)
  fir.call @foo4(%0#0) : (!fir.ref<!fir.logical<8>>) -> ()
  hlfir.end_associate %0#1, %0#2 : !fir.ref<!fir.logical<8>>, i1
  return
}
func.func private @foo4(!fir.ref<!fir.logical<8>>)
// CHECK-LABEL:   func.func @test_logical(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.logical<8>) {
// CHECK:  %[[VAL_1:.*]]:3 = hlfir.associate %[[VAL_0]] {uniq_name = "x"} : (!fir.logical<8>) -> (!fir.ref<!fir.logical<8>>, !fir.ref<!fir.logical<8>>, i1)
// CHECK:  fir.call @foo4(%[[VAL_1]]#0) : (!fir.ref<!fir.logical<8>>) -> ()
// CHECK:  hlfir.end_associate %[[VAL_1]]#1, %[[VAL_1]]#2 : !fir.ref<!fir.logical<8>>, i1


func.func @test_complex(%arg0: !fir.complex<8>) {
  %0:3 = hlfir.associate %arg0 {uniq_name = "x"} : (!fir.complex<8>) -> (!fir.ref<!fir.complex<8>>, !fir.ref<!fir.complex<8>>, i1)
  fir.call @foo5(%0#0) : (!fir.ref<!fir.complex<8>>) -> ()
  hlfir.end_associate %0#1, %0#2 : !fir.ref<!fir.complex<8>>, i1
  return
}
func.func private @foo5(!fir.ref<!fir.complex<8>>)
// CHECK-LABEL:   func.func @test_complex(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.complex<8>) {
// CHECK:  %[[VAL_1:.*]]:3 = hlfir.associate %[[VAL_0]] {uniq_name = "x"} : (!fir.complex<8>) -> (!fir.ref<!fir.complex<8>>, !fir.ref<!fir.complex<8>>, i1)
// CHECK:  fir.call @foo5(%[[VAL_1]]#0) : (!fir.ref<!fir.complex<8>>) -> ()
// CHECK:  hlfir.end_associate %[[VAL_1]]#1, %[[VAL_1]]#2 : !fir.ref<!fir.complex<8>>, i1


func.func @test_array(%arg0: !hlfir.expr<!fir.array<10x?xi32>>) {
  %shape = fir.undefined !fir.shape<2>
  %0:3 = hlfir.associate %arg0(%shape) {uniq_name = "x"} : (!hlfir.expr<!fir.array<10x?xi32>>, !fir.shape<2>) -> (!fir.box<!fir.array<10x?xi32>>, !fir.ref<!fir.array<10x?xi32>>, i1)
  fir.call @foo2(%0#0) : (!fir.box<!fir.array<10x?xi32>>) -> ()
  hlfir.end_associate %0#1, %0#2 : !fir.ref<!fir.array<10x?xi32>>, i1
  return
}
func.func private @foo6(!fir.box<!fir.array<10x?xi32>>)
// CHECK-LABEL:   func.func @test_array(
// CHECK-SAME:    %[[VAL_0:.*]]: !hlfir.expr<!fir.array<10x?xi32>>) {
// CHECK:  %[[VAL_1:.*]] = fir.undefined !fir.shape<2>
// CHECK:  %[[VAL_2:.*]]:3 = hlfir.associate %[[VAL_0]](%[[VAL_1]]) {uniq_name = "x"} : (!hlfir.expr<!fir.array<10x?xi32>>, !fir.shape<2>) -> (!fir.box<!fir.array<10x?xi32>>, !fir.ref<!fir.array<10x?xi32>>, i1)
// CHECK:  fir.call @foo2(%[[VAL_2]]#0) : (!fir.box<!fir.array<10x?xi32>>) -> ()
// CHECK:  hlfir.end_associate %[[VAL_2]]#1, %[[VAL_2]]#2 : !fir.ref<!fir.array<10x?xi32>>, i1
