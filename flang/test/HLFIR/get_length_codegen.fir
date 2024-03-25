// Test hlfir.get_length code generation
// RUN: fir-opt %s --bufferize-hlfir | FileCheck %s

func.func @_QPtest_char_get_length(%arg0: !fir.boxchar<1> {fir.bindc_name = "ch"}) -> index {
  %c3 = arith.constant 3 : index
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 {uniq_name = "_QFtest_char_get_lengthEch"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2 = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFtest_char_get_lengthEx"}
  %3:2 = hlfir.declare %2 {uniq_name = "_QFtest_char_get_lengthEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4 = fir.address_of(@_QQclX616263) : !fir.ref<!fir.char<1,3>>
  %5:2 = hlfir.declare %4 typeparams %c3 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX616263"} : (!fir.ref<!fir.char<1,3>>, index) -> (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>)
  %6 = arith.addi %0#1, %c3 : index
  %7 = hlfir.concat %5#0, %1#0 len %6 : (!fir.ref<!fir.char<1,3>>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
  %8:3 = hlfir.associate %7 typeparams %6 {uniq_name = ".tmp.assign"} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  %9 = hlfir.as_expr %8#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  %10 = hlfir.get_length %9 : (!hlfir.expr<!fir.char<1,?>>) -> index
  hlfir.destroy %9 : !hlfir.expr<!fir.char<1,?>>
  hlfir.end_associate %8#1, %8#2 : !fir.ref<!fir.char<1,?>>, i1
  return %10 : index
}
fir.global linkonce @_QQclX616263 constant : !fir.char<1,3> {
  %0 = fir.string_lit "abc"(3) : !fir.char<1,3>
  fir.has_value %0 : !fir.char<1,3>
}
// CHECK-LABEL:   func.func @_QPtest_char_get_length(
// CHECK-SAME:      %[[VAL_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "ch"}) -> index {
// CHECK:           %[[VAL_1:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_1]], %[[VAL_2]]#1 : index
// CHECK:           %[[VAL_33:.*]]:2 = hlfir.declare %[[VAL_31:.*]] typeparams %[[VAL_9]] {uniq_name = ".tmp"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           return %[[VAL_9]] : index
// CHECK:         }
