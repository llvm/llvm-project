! This test checks lowering of atomic and atomic update constructs with HLFIR
! RUN: bbc -hlfir -fopenacc -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenacc %s -o - | FileCheck %s

!CHECK-LABEL: @_QPsb
subroutine sb
!CHECK:   %[[W_REF:.*]] = fir.alloca i32 {bindc_name = "w", uniq_name = "_QFsbEw"}
!CHECK:   %[[W_DECL:.*]]:2 = hlfir.declare %[[W_REF]] {uniq_name = "_QFsbEw"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:   %[[X_REF:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFsbEx"}
!CHECK:   %[[X_DECL:.*]]:2 = hlfir.declare %[[X_REF]] {uniq_name = "_QFsbEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:   %[[Y_REF:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFsbEy"}
!CHECK:   %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y_REF]] {uniq_name = "_QFsbEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:   %[[Z_REF:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFsbEz"}
!CHECK:   %[[Z_DECL:.*]]:2 = hlfir.declare %[[Z_REF]] {uniq_name = "_QFsbEz"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  integer :: w, x, y, z

!CHECK:   %[[Y_VAL:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<i32>
!CHECK:   acc.atomic.update   %[[X_DECL]]#1 : !fir.ref<i32> {
!CHECK:   ^bb0(%[[ARG_X:.*]]: i32):
!CHECK:     %[[X_UPDATE_VAL:.*]] = arith.addi %[[ARG_X]], %[[Y_VAL]] : i32
!CHECK:     acc.yield %[[X_UPDATE_VAL]] : i32
!CHECK:   }
  !$acc atomic update
    x = x + y

!CHECK:   %[[Y_VAL:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<i32>
!CHECK:   acc.atomic.update %[[X_DECL]]#1 : !fir.ref<i32> {
!CHECK:   ^bb0(%[[ARG_X:.*]]: i32):
!CHECK:     %[[X_UPDATE_VAL:.*]] = arith.ori %[[ARG_X]], %[[Y_VAL]] : i32
!CHECK:     acc.yield %[[X_UPDATE_VAL]] : i32
!CHECK:   }
  !$acc atomic update
    x = ior(x,y)

!CHECK:   %[[W_VAL:.*]] = fir.load %[[W_DECL]]#0 : !fir.ref<i32>
!CHECK:   %[[X_VAL:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
!CHECK:   %[[Y_VAL:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<i32>
!CHECK:   acc.atomic.update %[[Z_DECL]]#1 : !fir.ref<i32> {
!CHECK:   ^bb0(%[[ARG_Z:.*]]: i32):
!CHECK:     %[[WX_CMP:.*]] = arith.cmpi slt, %[[W_VAL]], %[[X_VAL]] : i32
!CHECK:     %[[WX_MIN:.*]] = arith.select %[[WX_CMP]], %[[W_VAL]], %[[X_VAL]] : i32
!CHECK:     %[[WXY_CMP:.*]] = arith.cmpi slt, %[[WX_MIN]], %[[Y_VAL]] : i32
!CHECK:     %[[WXY_MIN:.*]] = arith.select %[[WXY_CMP]], %[[WX_MIN]], %[[Y_VAL]] : i32
!CHECK:     %[[WXYZ_CMP:.*]] = arith.cmpi slt, %[[WXY_MIN]], %[[ARG_Z]] : i32
!CHECK:     %[[WXYZ_MIN:.*]] = arith.select %[[WXYZ_CMP]], %[[WXY_MIN]], %[[ARG_Z]] : i32
!CHECK:     acc.yield %[[WXYZ_MIN]] : i32
!CHECK:   }
  !$acc atomic update
    z = min(w,x,y,z)

!CHECK:   return
end subroutine
