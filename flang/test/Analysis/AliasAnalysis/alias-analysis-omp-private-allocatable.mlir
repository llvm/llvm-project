// Use --mlir-disable-threading so that the AA queries are serialized
// as well as its diagnostic output.
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// Fortran code before simplification:
// SUBROUTINE mysub(ns,ne)
//      INTEGER  :: n
//      REAL(KIND=8), DIMENSION(:), allocatable     :: ar1
//      real(kind=8), dimension(20) :: ar2
//      REAL(KIND=8), DIMENSION(20) :: d
//
//!$OMP parallel PRIVATE(ar1)
//         d(1:1) = (/(DOT_PRODUCT(ar1(1:n), ar2(1:n)),n=1, 1)/)
//!$OMP END parallel
//   END SUBROUTINE

// CHECK-LABEL: Testing : "testPrivateAllocatable"
// CHECK: ar2#0 <-> ar1#0: NoAlias
// CHECK: ar2#1 <-> ar1#0: NoAlias
// CHECK: ar2#0 <-> ar1#1: NoAlias
// CHECK: ar2#1 <-> ar1#1: NoAlias

omp.private {type = private} @_QFmysubEar1_private_ref_box_heap_Uxf64 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>> alloc {
^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>):
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf64>>> {bindc_name = "ar1", pinned, uniq_name = "_QFmysubEar1"}
  %5:2 = hlfir.declare %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFmysubEar1"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>)
  omp.yield(%5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>)
} dealloc {
^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>):
  omp.yield
}
func.func @testPrivateAllocatable(%arg0: !fir.ref<i32> {fir.bindc_name = "ns"}, %arg1: !fir.ref<i32> {fir.bindc_name = "ne"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf64>>> {bindc_name = "ar1", uniq_name = "_QFmysubEar1"}
  %2 = fir.zero_bits !fir.heap<!fir.array<?xf64>>
  %c0 = arith.constant 0 : index
  %3 = fir.shape %c0 : (index) -> !fir.shape<1>
  %4 = fir.embox %2(%3) : (!fir.heap<!fir.array<?xf64>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf64>>>
  fir.store %4 to %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  %5:2 = hlfir.declare %1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFmysubEar1"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>)
  %c20 = arith.constant 20 : index
  %6 = fir.alloca !fir.array<20xf64> {bindc_name = "ar2", uniq_name = "_QFmysubEar2"}
  %7 = fir.shape %c20 : (index) -> !fir.shape<1>
  %8:2 = hlfir.declare %6(%7) {uniq_name = "_QFmysubEar2", test.ptr="ar2" } : (!fir.ref<!fir.array<20xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<20xf64>>, !fir.ref<!fir.array<20xf64>>)
  omp.parallel private(@_QFmysubEar1_private_ref_box_heap_Uxf64 %5#0 -> %arg2 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) {
    %20:2 = hlfir.declare %arg2 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFmysubEar1", test.ptr = "ar1"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>)
    omp.terminator
  }
  return
}
