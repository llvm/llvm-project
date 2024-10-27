// Use --mlir-disable-threading so that the AA queries are serialized
// as well as its diagnostic output.
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// Fortran source code:
//
// program TestAllocatableArray
// real(kind=8),  allocatable :: A(:)
// real(kind=8),  allocatable :: B(:)
// !$omp target
//    A(0) = B(0)
// !$omp end target
// end TestAllocatableArray

// CHECK-LABEL: Testing : "_QPTestAllocatableArray"
// CHECK-DAG: targetArrayB#0 <-> targetArrayA#0: NoAlias
func.func @_QPTestAllocatableArray() {
  %0 = fir.address_of(@_QFEa) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  %1:2 = hlfir.declare %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "ArrayA" } : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>)
  %2 = fir.address_of(@_QFEb) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  %3:2 = hlfir.declare %2 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "ArrayB" } : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>)
  %4 = fir.load %1#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %5 = fir.load %1#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  %c0_0 = arith.constant 0 : index
  %6:3 = fir.box_dims %5, %c0_0 : (!fir.box<!fir.heap<!fir.array<?xf64>>>, index) -> (index, index, index)
  %7:3 = fir.box_dims %4, %c0 : (!fir.box<!fir.heap<!fir.array<?xf64>>>, index) -> (index, index, index)
  %c0_1 = arith.constant 0 : index
  %8 = arith.subi %7#1, %c1 : index
  %9 = omp.map.bounds lower_bound(%c0_1 : index) upper_bound(%8 : index) extent(%7#1 : index) stride(%7#2 : index) start_idx(%6#0 : index) {stride_in_bytes = true}
  %10 = fir.box_offset %1#1 base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xf64>>>
  %11 = omp.map.info var_ptr(%1#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.array<?xf64>) var_ptr_ptr(%10 : !fir.llvm_ptr<!fir.ref<!fir.array<?xf64>>>) map_clauses(implicit, tofrom) capture(ByRef) bounds(%9) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xf64>>> {name = ""}
  %12 = omp.map.info var_ptr(%1#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.box<!fir.heap<!fir.array<?xf64>>>) map_clauses(implicit, tofrom) capture(ByRef) members(%11 : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xf64>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>> {name = "a"}
  %13 = fir.load %3#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  %c1_2 = arith.constant 1 : index
  %c0_3 = arith.constant 0 : index
  %14 = fir.load %3#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  %c0_4 = arith.constant 0 : index
  %15:3 = fir.box_dims %14, %c0_4 : (!fir.box<!fir.heap<!fir.array<?xf64>>>, index) -> (index, index, index)
  %16:3 = fir.box_dims %13, %c0_3 : (!fir.box<!fir.heap<!fir.array<?xf64>>>, index) -> (index, index, index)
  %c0_5 = arith.constant 0 : index
  %17 = arith.subi %16#1, %c1_2 : index
  %18 = omp.map.bounds lower_bound(%c0_5 : index) upper_bound(%17 : index) extent(%16#1 : index) stride(%16#2 : index) start_idx(%15#0 : index) {stride_in_bytes = true}
  %19 = fir.box_offset %3#1 base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xf64>>>
  %20 = omp.map.info var_ptr(%3#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.array<?xf64>) var_ptr_ptr(%19 : !fir.llvm_ptr<!fir.ref<!fir.array<?xf64>>>) map_clauses(implicit, tofrom) capture(ByRef) bounds(%18) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xf64>>> {name = ""}
  %21 = omp.map.info var_ptr(%3#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.box<!fir.heap<!fir.array<?xf64>>>) map_clauses(implicit, tofrom) capture(ByRef) members(%20 : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xf64>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>> {name = "b"}
  omp.target map_entries(%11 -> %arg0, %12 -> %arg1, %20 -> %arg2, %21 -> %arg3 : !fir.llvm_ptr<!fir.ref<!fir.array<?xf64>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xf64>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) {
    %22:2 = hlfir.declare %arg1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>)
    %23:2 = hlfir.declare %arg3 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEb"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>)
    %24 = fir.load %23#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
    %c0_6 = arith.constant 0 : index
    %25 = hlfir.designate %24 (%c0_6) {test.ptr = "targetArrayB"} : (!fir.box<!fir.heap<!fir.array<?xf64>>>, index) -> !fir.ref<f64>
    %26 = fir.load %25 : !fir.ref<f64>
    %27 = fir.load %22#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
    %c0_7 = arith.constant 0 : index
    %28 = hlfir.designate %27 (%c0_7) {test.ptr = "targetArrayA"} : (!fir.box<!fir.heap<!fir.array<?xf64>>>, index) -> !fir.ref<f64>
    hlfir.assign %26 to %28 : f64, !fir.ref<f64>
    omp.terminator
  }
  return
}
fir.global internal @_QFEa : !fir.box<!fir.heap<!fir.array<?xf64>>> {
}
fir.global internal @_QFEb : !fir.box<!fir.heap<!fir.array<?xf64>>> {
}
