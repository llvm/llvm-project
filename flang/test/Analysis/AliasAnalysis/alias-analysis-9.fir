// Use --mlir-disable-threading so that the AA queries are serialized
// as well as its diagnostic output.
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// Fortran source code:
// program TestTmpArrayAssignment
//   integer :: a, b, c
//   integer :: arrayD(3)
//
//   arrayD = [ a, b, c ]
// end program TestTmpArrayAssignment

// CHECK-LABEL: Testing : "_QPTestTmpArrayAssignment"
// CHECK-DAG: ArrayD#0 <-> tmp_array#0: NoAlias
func.func @_QPTestTmpArrayAssignment() attributes {fir.bindc_name = "testtmparrayassignment"} {
  %0 = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFEa"}
  %1:2 = hlfir.declare %0 {uniq_name = "_QFEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %c3 = arith.constant 3 : index
  %2 = fir.alloca !fir.array<3xi32> {bindc_name = "arrayd", uniq_name = "_QFEarrayd", test.ptr = "ArrayD" }
  %3 = fir.shape %c3 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %2(%3) {uniq_name = "_QFEarrayd"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
  %5 = fir.alloca i32 {bindc_name = "b", uniq_name = "_QFEb"}
  %6:2 = hlfir.declare %5 {uniq_name = "_QFEb"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7 = fir.alloca i32 {bindc_name = "c", uniq_name = "_QFEc"}
  %8:2 = hlfir.declare %7 {uniq_name = "_QFEc"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %c3_0 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %c1_1 = arith.constant 1 : index
  %9 = fir.allocmem !fir.array<3xi32> {bindc_name = ".tmp.arrayctor", uniq_name = ""}
  %10 = fir.shape %c3_0 : (index) -> !fir.shape<1>
  %11:2 = hlfir.declare %9(%10) {uniq_name = ".tmp.arrayctor"} : (!fir.heap<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<3xi32>>, !fir.heap<!fir.array<3xi32>>)
  %12 = fir.load %1#0 : !fir.ref<i32>
  %13 = arith.addi %c1, %c1_1 : index
  %14 = hlfir.designate %11#0 (%c1)  : (!fir.heap<!fir.array<3xi32>>, index) -> !fir.ref<i32>
  hlfir.assign %12 to %14 : i32, !fir.ref<i32>
  %15 = fir.load %6#0 : !fir.ref<i32>
  %16 = arith.addi %13, %c1_1 : index
  %17 = hlfir.designate %11#0 (%13)  : (!fir.heap<!fir.array<3xi32>>, index) -> !fir.ref<i32>
  hlfir.assign %15 to %17 : i32, !fir.ref<i32>
  %18 = fir.load %8#0 : !fir.ref<i32>
  %19 = hlfir.designate %11#0 (%16)  : (!fir.heap<!fir.array<3xi32>>, index) -> !fir.ref<i32>
  hlfir.assign %18 to %19 : i32, !fir.ref<i32>
  %true = arith.constant true
  %20 = hlfir.as_expr %11#0 move %true {test.ptr = "tmp_array"}: (!fir.heap<!fir.array<3xi32>>, i1) -> !hlfir.expr<3xi32>
  hlfir.assign %20 to %4#0 : !hlfir.expr<3xi32>, !fir.ref<!fir.array<3xi32>>
  hlfir.destroy %20 : !hlfir.expr<3xi32>
  return
}
