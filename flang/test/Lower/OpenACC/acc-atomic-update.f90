! This test checks lowering of atomic and atomic update constructs
! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenacc -emit-hlfir %s -o - | FileCheck %s

program acc_atomic_update_test
    integer :: x, y, z
    integer, pointer :: a, b
    integer, target :: c, d
    integer(1) :: i1

    a=>c
    b=>d

!CHECK: %[[A:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "a", uniq_name = "_QFEa"}
!CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK: %[[B:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "b", uniq_name = "_QFEb"}
!CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[B]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFEb"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK: %[[C_ADDR:.*]] = fir.address_of(@_QFEc) : !fir.ref<i32>
!CHECK: %[[D_ADDR:.*]] = fir.address_of(@_QFEd) : !fir.ref<i32>
!CHECK: %[[I1:.*]] = fir.alloca i8 {bindc_name = "i1", uniq_name = "_QFEi1"}
!CHECK: %[[I1_DECL:.*]]:2 = hlfir.declare %[[I1]] {uniq_name = "_QFEi1"} : (!fir.ref<i8>) -> (!fir.ref<i8>, !fir.ref<i8>)
!CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Z:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFEz"}
!CHECK: %[[Z_DECL:.*]]:2 = hlfir.declare %[[Z]] {uniq_name = "_QFEz"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[LOAD_A:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[BOX_ADDR_A:.*]] = fir.box_addr %[[LOAD_A]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: %[[LOAD_B:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[BOX_ADDR_B:.*]] = fir.box_addr %[[LOAD_B]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: %[[LOAD_BOX_ADDR_B:.*]] = fir.load %[[BOX_ADDR_B]] : !fir.ptr<i32>
!CHECK: acc.atomic.update %[[BOX_ADDR_A]] : !fir.ptr<i32> {
!CHECK: ^bb0(%[[ARG0:.*]]: i32):
!CHECK:   %[[ADD:.*]] = arith.addi %[[ARG0]], %[[LOAD_BOX_ADDR_B]] : i32
!CHECK:   acc.yield %[[ADD]] : i32
!CHECK: }

    !$acc atomic update
        a = a + b 

!CHECK: {{.*}} = arith.constant 1 : i32
!CHECK: acc.atomic.update %[[Y_DECL]]#1 : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    %[[RESULT:.*]] = arith.addi %[[ARG]], {{.*}} : i32
!CHECK:    acc.yield %[[RESULT]] : i32
!CHECK:  }
!CHECK:  %[[LOADED_X:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
!CHECK:  acc.atomic.update %[[Z_DECL]]#1 : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    %[[RESULT:.*]] = arith.muli %[[LOADED_X]], %[[ARG]] : i32
!CHECK:    acc.yield %[[RESULT]] : i32
!CHECK:  }
    !$acc atomic 
        y = y + 1
    !$acc atomic update
        z = x * z 

!CHECK:  %[[C1_VAL:.*]] = arith.constant 1 : i32
!CHECK:  acc.atomic.update %[[I1_DECL]]#1 : !fir.ref<i8> {
!CHECK:  ^bb0(%[[VAL:.*]]: i8):
!CHECK:    %[[CVT_VAL:.*]] = fir.convert %[[VAL]] : (i8) -> i32
!CHECK:    %[[ADD_VAL:.*]] = arith.addi %[[CVT_VAL]], %[[C1_VAL]] : i32
!CHECK:    %[[UPDATED_VAL:.*]] = fir.convert %[[ADD_VAL]] : (i32) -> i8
!CHECK:    acc.yield %[[UPDATED_VAL]] : i8
!CHECK:  }
    !$acc atomic
      i1 = i1 + 1
    !$acc end atomic
!CHECK:  return
!CHECK: }
end program acc_atomic_update_test

