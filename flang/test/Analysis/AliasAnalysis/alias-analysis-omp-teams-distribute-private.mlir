// Use --mlir-disable-threading so that the AA queries are serialized
// as well as its diagnostic output.
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// Fortran code:
//
// program main
// integer :: arrayA(10,10)
// integer :: tmp(2)
// integer :: i,j
// !$omp teams distribute parallel do private(tmp)
// do j = 1, 10
//   do i = 1,10
//     tmp = [i,j]
//     arrayA = tmp(1)
//   end do
// end do
// end program main

// CHECK-LABEL: Testing : "_QQmain"
// CHECK-DAG: tmp_private_array#0 <-> unnamed_array#0: NoAlias
// CHECK-DAG: tmp_private_array#1 <-> unnamed_array#0: NoAlias

omp.private {type = private} @_QFEi_private_ref_i32 : i32
omp.private {type = private} @_QFEj_private_ref_i32 : i32
omp.private {type = private} @_QFEtmp_private_ref_2xi32 : !fir.array<2xi32>

func.func @_QQmain() attributes {fir.bindc_name = "main"} {
  %0 = fir.address_of(@_QFEarraya) : !fir.ref<!fir.array<10x10xi32>>
  %c10 = arith.constant 10 : index
  %c10_0 = arith.constant 10 : index
  %1 = fir.shape %c10, %c10_0 : (index, index) -> !fir.shape<2>
  %2:2 = hlfir.declare %0(%1) {uniq_name = "_QFEarraya"} : (!fir.ref<!fir.array<10x10xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<10x10xi32>>, !fir.ref<!fir.array<10x10xi32>>)
  %3 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFEj"}
  %6:2 = hlfir.declare %5 {uniq_name = "_QFEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %c2 = arith.constant 2 : index
  %7 = fir.alloca !fir.array<2xi32> {bindc_name = "tmp", uniq_name = "_QFEtmp"}
  %8 = fir.shape %c2 : (index) -> !fir.shape<1>
  %9:2 = hlfir.declare %7(%8) {uniq_name = "_QFEtmp"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
  omp.teams {
    omp.parallel private(@_QFEtmp_private_ref_2xi32 %9#0 -> %arg0, @_QFEj_private_ref_i32 %6#0 -> %arg1, @_QFEi_private_ref_i32 %4#0 -> %arg2 : !fir.ref<!fir.array<2xi32>>, !fir.ref<i32>, !fir.ref<i32>) {
      %c2_1 = arith.constant 2 : index
      %10 = fir.shape %c2_1 : (index) -> !fir.shape<1>
      %11:2 = hlfir.declare %arg0(%10) {uniq_name = "_QFEtmp", test.ptr = "tmp_private_array"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
      %12:2 = hlfir.declare %arg1 {uniq_name = "_QFEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %13:2 = hlfir.declare %arg2 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %c1_i32 = arith.constant 1 : i32
      %c10_i32 = arith.constant 10 : i32
      %c1_i32_2 = arith.constant 1 : i32
      omp.distribute {
        omp.wsloop {
          omp.loop_nest (%arg3) : i32 = (%c1_i32) to (%c10_i32) inclusive step (%c1_i32_2) {
            fir.store %arg3 to %12#1 : !fir.ref<i32>
            %c1_i32_3 = arith.constant 1 : i32
            %14 = fir.convert %c1_i32_3 : (i32) -> index
            %c10_i32_4 = arith.constant 10 : i32
            %15 = fir.convert %c10_i32_4 : (i32) -> index
            %c1 = arith.constant 1 : index
            %16 = fir.convert %14 : (index) -> i32
            %17:2 = fir.do_loop %arg4 = %14 to %15 step %c1 iter_args(%arg5 = %16) -> (index, i32) {
              fir.store %arg5 to %13#1 : !fir.ref<i32>
              %c2_5 = arith.constant 2 : index
              %c1_6 = arith.constant 1 : index
              %c1_7 = arith.constant 1 : index
              %18 = fir.allocmem !fir.array<2xi32> {bindc_name = ".tmp.arrayctor", uniq_name = ""}
              %19 = fir.shape %c2_5 : (index) -> !fir.shape<1>
              %20:2 = hlfir.declare %18(%19) {uniq_name = ".tmp.arrayctor"} : (!fir.heap<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<2xi32>>, !fir.heap<!fir.array<2xi32>>)
              %21 = fir.load %13#0 : !fir.ref<i32>
              %22 = arith.addi %c1_6, %c1_7 : index
              %23 = hlfir.designate %20#0 (%c1_6)  : (!fir.heap<!fir.array<2xi32>>, index) -> !fir.ref<i32>
              hlfir.assign %21 to %23 : i32, !fir.ref<i32>
              %24 = fir.load %12#0 : !fir.ref<i32>
              %25 = hlfir.designate %20#0 (%22)  : (!fir.heap<!fir.array<2xi32>>, index) -> !fir.ref<i32>
              hlfir.assign %24 to %25 : i32, !fir.ref<i32>
              %true = arith.constant true
              %26 = hlfir.as_expr %20#0 move %true {test.ptr = "unnamed_array"} : (!fir.heap<!fir.array<2xi32>>, i1) -> !hlfir.expr<2xi32>
              hlfir.assign %26 to %11#0 : !hlfir.expr<2xi32>, !fir.ref<!fir.array<2xi32>>
              hlfir.destroy %26 : !hlfir.expr<2xi32>
              %c1_8 = arith.constant 1 : index
              %27 = hlfir.designate %11#0 (%c1_8)  : (!fir.ref<!fir.array<2xi32>>, index) -> !fir.ref<i32>
              %28 = fir.load %27 : !fir.ref<i32>
              hlfir.assign %28 to %2#0 : i32, !fir.ref<!fir.array<10x10xi32>>
              %29 = arith.addi %arg4, %c1 : index
              %30 = fir.convert %c1 : (index) -> i32
              %31 = fir.load %13#1 : !fir.ref<i32>
              %32 = arith.addi %31, %30 : i32
              fir.result %29, %32 : index, i32
            }
            fir.store %17#1 to %13#1 : !fir.ref<i32>
            omp.yield
          }
        } {omp.composite}
      } {omp.composite}
      omp.terminator
    } {omp.composite}
    omp.terminator
  }
  return
}
fir.global internal @_QFEarraya : !fir.array<10x10xi32> {
  %0 = fir.zero_bits !fir.array<10x10xi32>
  fir.has_value %0 : !fir.array<10x10xi32>
}
