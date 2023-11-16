! This test checks lowering of atomic and atomic update constructs
! RUN: bbc -fopenacc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -fopenacc %s -o - | FileCheck %s

program acc_atomic_update_test
    integer :: x, y, z
    integer, pointer :: a, b
    integer, target :: c, d
    integer(1) :: i1

    a=>c
    b=>d

!CHECK: %[[A:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "a", uniq_name = "_QFEa"}
!CHECK: %[[B:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "b", uniq_name = "_QFEb"}
!CHECK: %[[C_ADDR:.*]] = fir.address_of(@_QFEc) : !fir.ref<i32>
!CHECK: %[[D_ADDR:.*]] = fir.address_of(@_QFEd) : !fir.ref<i32>
!CHECK: %[[I1:.*]] = fir.alloca i8 {bindc_name = "i1", uniq_name = "_QFEi1"}
!CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK: %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK: %[[Z:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFEz"}
!CHECK: %[[LOAD_A:.*]] = fir.load %[[A]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[BOX_ADDR_A:.*]] = fir.box_addr %[[LOAD_A]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: %[[LOAD_B:.*]] = fir.load %[[B]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
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
!CHECK: acc.atomic.update   %[[Y]] : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    %[[RESULT:.*]] = arith.addi %[[ARG]], {{.*}} : i32
!CHECK:    acc.yield %[[RESULT]] : i32
!CHECK:  }
!CHECK:  %[[LOADED_X:.*]] = fir.load %[[X]] : !fir.ref<i32>
!CHECK:  acc.atomic.update   %[[Z]] : !fir.ref<i32> {
!CHECK:  ^bb0(%[[ARG:.*]]: i32):
!CHECK:    %[[RESULT:.*]] = arith.muli %[[LOADED_X]], %[[ARG]] : i32
!CHECK:    acc.yield %[[RESULT]] : i32
!CHECK:  }
    !$acc atomic 
        y = y + 1
    !$acc atomic update
        z = x * z 

!CHECK:  %[[C1_VAL:.*]] = arith.constant 1 : i32
!CHECK:  acc.atomic.update   %[[I1]] : !fir.ref<i8> {
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

