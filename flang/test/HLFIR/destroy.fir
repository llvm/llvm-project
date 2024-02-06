// Test hlfir.destroy operation parse, verify (no errors), and unparse.

// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @test(%expr : !hlfir.expr<?x?xf64>) {
  hlfir.destroy %expr : !hlfir.expr<?x?xf64>
  return
}
// CHECK-LABEL: func.func @test(
// CHECK-SAME:    %[[VAL_0:.*]]: !hlfir.expr<?x?xf64>) {
// CHECK:   hlfir.destroy %[[VAL_0]] : !hlfir.expr<?x?xf64>

func.func @test_finalize_dt(%expr : !hlfir.expr<?x!fir.type<_QMtypesTt>>) {
  hlfir.destroy %expr finalize : !hlfir.expr<?x!fir.type<_QMtypesTt>>
  return
}
// CHECK-LABEL:   func.func @test_finalize_dt(
// CHECK-SAME:        %[[VAL_0:.*]]: !hlfir.expr<?x!fir.type<_QMtypesTt>>) {
// CHECK:           hlfir.destroy %[[VAL_0]] finalize : !hlfir.expr<?x!fir.type<_QMtypesTt>>

func.func @test_finalize_poly(%expr : !hlfir.expr<?x!fir.type<_QMtypesTt>?>) {
  hlfir.destroy %expr finalize : !hlfir.expr<?x!fir.type<_QMtypesTt>?>
  return
}
// CHECK-LABEL:   func.func @test_finalize_poly(
// CHECK-SAME:        %[[VAL_0:.*]]: !hlfir.expr<?x!fir.type<_QMtypesTt>?>) {
// CHECK:           hlfir.destroy %[[VAL_0]] finalize : !hlfir.expr<?x!fir.type<_QMtypesTt>?>
