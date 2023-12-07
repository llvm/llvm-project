! This test checks lowering of atomic and atomic update constructs with HLFIR
! RUN: bbc -hlfir -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -flang-experimental-hlfir -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine sb
  integer :: x, y

  !$omp atomic update
    x = x + y
end subroutine

!CHECK-LABEL: @_QPsb
!CHECK:   %[[X_REF:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFsbEx"}
!CHECK:   %[[X_DECL:.*]]:2 = hlfir.declare %[[X_REF]] {uniq_name = "_QFsbEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:   %[[Y_REF:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFsbEy"}
!CHECK:   %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y_REF]] {uniq_name = "_QFsbEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:   omp.atomic.update   %[[X_DECL]]#0 : !fir.ref<i32> {
!CHECK:   ^bb0(%[[ARG_X:.*]]: i32):
!CHECK:     %[[Y_VAL:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<i32>
!CHECK:     %[[X_UPDATE_VAL:.*]] = arith.addi %[[ARG_X]], %[[Y_VAL]] : i32
!CHECK:     omp.yield(%[[X_UPDATE_VAL]] : i32)
!CHECK:   }
!CHECK:   return
