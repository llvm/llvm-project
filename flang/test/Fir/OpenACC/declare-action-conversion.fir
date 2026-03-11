// RUN: fir-opt %s --acc-declare-action-conversion -split-input-file -o - | FileCheck %s

// Check declare_action conversion for global variables
// module mm
// contains
//   subroutine sub()
//     real :: arr(10)
//     !$acc declare create(arr)
//     arr(1) = 3.0
//   end subroutine sub
// end module mm
//
// program main
//   use mm
//   call sub()
// end program main
module {
  fir.global @_QMmmEarr {acc.declare = #acc.declare<dataClause =  acc_copyin>} : !fir.box<!fir.heap<!fir.array<?xf32>>> {
  }
  func.func private @_QMmmEarr_acc_declare_update_desc_post_alloc() {
    return
  }
// CHECK: func.func private @_QMmmEarr_acc_declare_update_desc_post_alloc() attributes {acc.declare_action}
// CHECK-LABEL: func.func @_QMmmPsub
  func.func @_QMmmPsub() {
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@_QMmmEarr) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %2 = fir.declare %1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMmmEarr"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %3 = fir.allocmem !fir.array<?xf32>, %c100 {fir.must_be_heap = true, uniq_name = "_QMmmEarr.alloc"}
    %4 = fir.shape %c100 : (index) -> !fir.shape<1>
    %5 = fir.embox %3(%4) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
    fir.store %5 to %2 {acc.declare_action = #acc.declare_action<postAlloc = @_QMmmEarr_acc_declare_update_desc_post_alloc>} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    return
  }
// CHECK: fir.store
// CHECK: fir.call @_QMmmEarr_acc_declare_update_desc_post_alloc
}

// -----

// Test declare_action on runtime allocation path (fir.call) with box<none> argument.
module {
  func.func private @_QFdeclareAp_acc_declare_post_alloc(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) {
    return
  }
  func.func @_QPdeclare() {
    %1 = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>> {uniq_name = "_QFdeclareEap"}
    %2 = fir.declare %1 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFdeclareEap"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    %3 = fir.convert %2 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
    %f = arith.constant 0 : i1
    %absent = fir.absent !fir.box<none>
    %s_i8 = fir.zero_bits !fir.ref<i8>
    %c4_i32 = arith.constant 4 : i32
    fir.call @_FortranAPointerAllocate(%3, %f, %absent, %s_i8, %c4_i32) {acc.declare_action = #acc.declare_action<postAlloc = @_QFdeclareAp_acc_declare_post_alloc>} : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
    return
  }
// CHECK-LABEL: func.func @_QPdeclare
// CHECK: %[[DECL:.*]] = fir.declare %{{.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
// CHECK: %[[CVT:.*]] = fir.convert %[[DECL]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK: fir.call @_FortranAPointerAllocate(%[[CVT]],
// CHECK-NEXT: fir.call @_QFdeclareAp_acc_declare_post_alloc(%[[DECL]])
  func.func private @_FortranAPointerAllocate(!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
}

// -----

// Test structured declare_action (allocatable with postAlloc)
module {
 func.func private @_QMmmFsubEarr_acc_declare_update_desc_post_alloc(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
   return
 }
 func.func private @_QMmmFsubEarr_acc_declare_update_desc_pre_dealloc(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
   return
 }
 func.func private @_QMmmFsubEarr_acc_declare_update_desc_post_dealloc(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
   return
 }
// CHECK: func.func private @_QMmmFsubEarr_acc_declare_update_desc_post_alloc(%{{.*}}) attributes {acc.declare_action}
// CHECK-LABEL: func.func @_QMmmPsub
  func.func @_QMmmPsub() {
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %1 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "arr", uniq_name = "_QMmmFsubEarr"}
    %2 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
    %3 = fir.shape %c0 : (index) -> !fir.shape<1>
    %4 = fir.embox %2(%3) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
    %5 = fir.declare %1 {acc.declare = #acc.declare<dataClause =  acc_copyin>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMmmFsubEarr"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %8 = fir.allocmem !fir.array<?xf32>, %c100 {fir.must_be_heap = true, uniq_name = "_QMmmFsubEarr.alloc"}
    %9 = fir.shape %c100 : (index) -> !fir.shape<1>
    %10 = fir.embox %8(%9) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
    fir.store %10 to %5 {acc.declare_action = #acc.declare_action<postAlloc = @_QMmmFsubEarr_acc_declare_update_desc_post_alloc>} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    return
  }
// CHECK: %[[DECLARE:.*]] = fir.declare {{.*}}
// CHECK: fir.store
// CHECK: fir.call @_QMmmFsubEarr_acc_declare_update_desc_post_alloc(%[[DECLARE]])
}
