// RUN: fir-opt --boxed-procedure %s | FileCheck %s
// Test the boxed procedure pass with OpenMP declaration operations.
// Check minimally, only arguments, yields and the private types.

// Test a private declaration with one region (alloc)
//CHECK: omp.private {type = private}  @_QFsub1Et1_private_ref_rec__QFsub1Tt : !fir.ref<!fir.type<_QFsub1TtUnboxProc{p1:() -> ()}>> alloc {
omp.private {type = private} @_QFsub1Et1_private_ref_rec__QFsub1Tt : !fir.ref<!fir.type<_QFsub1Tt{p1:!fir.boxproc<() -> ()>}>> alloc {
//CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.type<_QFsub1TtUnboxProc{p1:() -> ()}>>):
^bb0(%arg0: !fir.ref<!fir.type<_QFsub1Tt{p1:!fir.boxproc<() -> ()>}>>):
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.alloca !fir.type<_QFsub1Tt{p1:!fir.boxproc<() -> ()>}> {bindc_name = "t1", pinned, uniq_name = "_QFsub1Et1"}
  %1 = fir.declare %0 {uniq_name = "_QFsub1Et1"} : (!fir.ref<!fir.type<_QFsub1Tt{p1:!fir.boxproc<() -> ()>}>>) -> !fir.ref<!fir.type<_QFsub1Tt{p1:!fir.boxproc<() -> ()>}>>
  %2 = fir.embox %1 : (!fir.ref<!fir.type<_QFsub1Tt{p1:!fir.boxproc<() -> ()>}>>) -> !fir.box<!fir.type<_QFsub1Tt{p1:!fir.boxproc<() -> ()>}>>
  %3 = fir.address_of(@_QQclXea6256ba131ddd9c2210e68030a0edd3) : !fir.ref<!fir.char<1,49>>
  %4 = fir.convert %2 : (!fir.box<!fir.type<_QFsub1Tt{p1:!fir.boxproc<() -> ()>}>>) -> !fir.box<none>
  %5 = fir.convert %3 : (!fir.ref<!fir.char<1,49>>) -> !fir.ref<i8>
  %6 = fir.call @_FortranAInitialize(%4, %5, %c1_i32) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> none
//CHECK: omp.yield(%{{.*}} : !fir.ref<!fir.type<_QFsub1TtUnboxProc{p1:() -> ()}>>)
  omp.yield(%1 : !fir.ref<!fir.type<_QFsub1Tt{p1:!fir.boxproc<() -> ()>}>>)
}
func.func @_QPsub1() {
  %0 = fir.alloca !fir.type<_QFsub1Tt{p1:!fir.boxproc<() -> ()>}> {bindc_name = "t1", uniq_name = "_QFsub1Et1"}
  %1 = fir.declare %0 {uniq_name = "_QFsub1Et1"} : (!fir.ref<!fir.type<_QFsub1Tt{p1:!fir.boxproc<() -> ()>}>>) -> !fir.ref<!fir.type<_QFsub1Tt{p1:!fir.boxproc<() -> ()>}>>
//CHECK: omp.parallel private(@_QFsub1Et1_private_ref_rec__QFsub1Tt %{{.*}} -> %{{.*}} : !fir.ref<!fir.type<_QFsub1TtUnboxProc{p1:() -> ()}>>) {
  omp.parallel private(@_QFsub1Et1_private_ref_rec__QFsub1Tt %1 -> %arg0 : !fir.ref<!fir.type<_QFsub1Tt{p1:!fir.boxproc<() -> ()>}>>) {
    %2 = fir.declare %arg0 {uniq_name = "_QFsub1Et1"} : (!fir.ref<!fir.type<_QFsub1Tt{p1:!fir.boxproc<() -> ()>}>>) -> !fir.ref<!fir.type<_QFsub1Tt{p1:!fir.boxproc<() -> ()>}>>
    omp.terminator
  }
  return
}


// Test a private declaration with all regions (alloc, copy, dealloc)
//CHECK: omp.private {type = firstprivate} @_QFsub2Et1_firstprivate_ref_box_heap_rec__QFsub2Tt : 
//CHECK-SAME: !fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2TtUnboxProc{p1:() -> ()}>>>> alloc {
omp.private {type = firstprivate} @_QFsub2Et1_firstprivate_ref_box_heap_rec__QFsub2Tt : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>>> alloc {
//CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2TtUnboxProc{p1:() -> ()}>>>>):
^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>>>):
  %0 = fir.alloca !fir.box<!fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>> {bindc_name = "t1", pinned, uniq_name = "_QFsub2Et1"}
  %1 = fir.declare %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub2Et1"} : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>>>
//CHECK: omp.yield(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2TtUnboxProc{p1:() -> ()}>>>>)
  omp.yield(%1 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>>>)
} copy {
//CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2TtUnboxProc{p1:() -> ()}>>>>,
//CHECK-SAME: %{{.*}}: !fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2TtUnboxProc{p1:() -> ()}>>>>):
^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>>>):
  %c5_i32 = arith.constant 5 : i32
  %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>>>
  %1 = fir.box_addr %0 : (!fir.box<!fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>>) -> !fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>
  %2 = fir.embox %1 : (!fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>) -> !fir.box<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>
  %3 = fir.address_of(@_QQclXea) : !fir.ref<!fir.char<1,8>>
  %4 = fir.convert %arg1 : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>>>) -> !fir.ref<!fir.box<none>>
  %5 = fir.convert %2 : (!fir.box<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>) -> !fir.box<none>
  %6 = fir.convert %3 : (!fir.ref<!fir.char<1,8>>) -> !fir.ref<i8>
  %7 = fir.call @_FortranAAssign(%4, %5, %6, %c5_i32) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
//CHECK: omp.yield(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2TtUnboxProc{p1:() -> ()}>>>>)
  omp.yield(%arg1 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>>>)
} dealloc {
//CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2TtUnboxProc{p1:() -> ()}>>>>):
^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>>>):
  %c5_i32 = arith.constant 5 : i32
  %false = arith.constant false
  %0 = fir.absent !fir.box<none>
  %1 = fir.address_of(@_QQclXea) : !fir.ref<!fir.char<1,8>>
  %2 = fir.convert %arg0 : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>>>) -> !fir.ref<!fir.box<none>>
  %3 = fir.convert %1 : (!fir.ref<!fir.char<1,8>>) -> !fir.ref<i8>
  %4 = fir.call @_FortranAAllocatableDeallocate(%2, %false, %0, %3, %c5_i32) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
  omp.yield
}
func.func @_QPsub2() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>> {bindc_name = "t1", uniq_name = "_QFsub2Et1"}
  %1 = fir.declare %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub2Et1"} : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>>>
//CHECK: omp.parallel private(@_QFsub2Et1_firstprivate_ref_box_heap_rec__QFsub2Tt %{{.*}} -> %{{.*}} :
//CHECK-SAME: !fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2TtUnboxProc{p1:() -> ()}>>>>) {
  omp.parallel private(@_QFsub2Et1_firstprivate_ref_box_heap_rec__QFsub2Tt %1 -> %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>>>) {
    %2 = fir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub2Et1"} : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.type<_QFsub2Tt{p1:!fir.boxproc<() -> ()>}>>>>
    omp.terminator
  }
  return
}
func.func private @_FortranAInitialize(!fir.box<none>, !fir.ref<i8>, i32) -> none attributes {fir.runtime}
fir.global linkonce @_QQclXea constant : !fir.char<1,8> {
  %0 = fir.string_lit "pp.f90\00"(8) : !fir.char<1,8>
  fir.has_value %0 : !fir.char<1,8>
}
func.func private @_FortranAAllocatableDeallocate(!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32 attributes {fir.runtime}
func.func private @_FortranAAssign(!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none attributes {fir.runtime}
