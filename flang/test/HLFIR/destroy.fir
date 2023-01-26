// Test hlfir.destroy operation parse, verify (no errors), and unparse.

// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @test(%expr : !hlfir.expr<?x?xf64>) {
  hlfir.destroy %expr : !hlfir.expr<?x?xf64>
  return
}
// CHECK-LABEL: func.func @test(
// CHECK-SAME:    %[[VAL_0:.*]]: !hlfir.expr<?x?xf64>) {
// CHECK:   hlfir.destroy %[[VAL_0]] : !hlfir.expr<?x?xf64>
