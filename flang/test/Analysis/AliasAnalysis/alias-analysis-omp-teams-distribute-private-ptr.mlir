// Use --mlir-disable-threading so that the AA queries are serialized
// as well as its diagnostic output.
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// Fortran code:
// program main
// integer, target :: arrayA(10)
// integer, pointer, dimension(:) :: ptrA
// integer :: i
// ptrA => arrayA
// !$omp teams distribute parallel do firstprivate(ptrA)
// do i = 1, 10
//   arrayA(i) = arrayA(i) + ptrA(i);
// end do
// end program main

// CHECK-LABEL: Testing : "_QQmain"
// CHECK-DAG:   ptrA#0 <-> ArrayA#0: MayAlias

omp.private {type = private} @_QFEi_private_ref_i32 : !fir.ref<i32> alloc {
^bb0(%arg0: !fir.ref<i32>):
  %0 = fir.alloca i32 {bindc_name = "i", pinned, uniq_name = "_QFEi"}
  %1:2 = hlfir.declare %0 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  omp.yield(%1#0 : !fir.ref<i32>)
}
omp.private {type = firstprivate} @_QFEptra_firstprivate_ref_box_ptr_Uxi32 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> alloc {
^bb0(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>):
  %0 = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "ptra", pinned, uniq_name = "_QFEptra"}
  %1:2 = hlfir.declare %0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFEptra"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
  omp.yield(%1#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
} copy {
^bb0(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>):
  %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  fir.store %0 to %arg1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  omp.yield(%arg1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
}
func.func @_QQmain() attributes {fir.bindc_name = "main"} {
  %0 = fir.address_of(@_QFEarraya) : !fir.ref<!fir.array<10xi32>>
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %0(%1) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFEarraya"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  %3 = fir.address_of(@_QFEarrayb) : !fir.ref<!fir.array<10xi32>>
  %c10_0 = arith.constant 10 : index
  %4 = fir.shape %c10_0 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %3(%4) {uniq_name = "_QFEarrayb"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  %6 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  %7:2 = hlfir.declare %6 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %8 = fir.address_of(@_QFEptra) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %9:2 = hlfir.declare %8 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFEptra"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
  %10 = fir.shape %c10 : (index) -> !fir.shape<1>
  %11 = fir.embox %2#1(%10) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
  fir.store %11 to %9#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  omp.teams {
    omp.parallel private(@_QFEptra_firstprivate_ref_box_ptr_Uxi32 %9#0 -> %arg0, @_QFEi_private_ref_i32 %7#0 -> %arg1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<i32>) {
      %12:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFEptra"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
      %13:2 = hlfir.declare %arg1 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %c1_i32 = arith.constant 1 : i32
      %c10_i32 = arith.constant 10 : i32
      %c1_i32_1 = arith.constant 1 : i32
      omp.distribute {
        omp.wsloop {
          omp.loop_nest (%arg2) : i32 = (%c1_i32) to (%c10_i32) inclusive step (%c1_i32_1) {
            fir.store %arg2 to %13#1 : !fir.ref<i32>
            %14 = fir.load %13#0 : !fir.ref<i32>
            %15 = fir.convert %14 : (i32) -> i64
            %16 = hlfir.designate %2#0 (%15)  : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
            %17 = fir.load %16 : !fir.ref<i32>
            %18 = fir.load %12#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
            %19 = fir.load %13#0 : !fir.ref<i32>
            %20 = fir.convert %19 : (i32) -> i64
            %21 = hlfir.designate %18 (%20) {test.ptr = "ptrA" } : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, i64) -> !fir.ref<i32>
            %22 = fir.load %21 : !fir.ref<i32>
            %23 = arith.addi %17, %22 : i32
            %24 = fir.load %13#0 : !fir.ref<i32>
            %25 = fir.convert %24 : (i32) -> i64
            %26 = hlfir.designate %2#0 (%25) {test.ptr = "ArrayA"}  : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
            hlfir.assign %23 to %26 : i32, !fir.ref<i32>
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
fir.global internal @_QFEarraya target : !fir.array<10xi32> {
  %0 = fir.zero_bits !fir.array<10xi32>
  fir.has_value %0 : !fir.array<10xi32>
}
fir.global internal @_QFEarrayb : !fir.array<10xi32> {
  %0 = fir.zero_bits !fir.array<10xi32>
  fir.has_value %0 : !fir.array<10xi32>
}
fir.global internal @_QFEptra : !fir.box<!fir.ptr<!fir.array<?xi32>>> {
  %0 = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
  %c0 = arith.constant 0 : index
  %1 = fir.shape %c0 : (index) -> !fir.shape<1>
  %2 = fir.embox %0(%1) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
  fir.has_value %2 : !fir.box<!fir.ptr<!fir.array<?xi32>>>
}
