! This test checks lowering of sequential loops in OpenMP parallel.
! The loop indices of these loops should be privatised.

! RUN: bbc -hlfir -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -flang-experimental-hlfir -fopenmp %s -o - | FileCheck %s


subroutine sb1
  integer i
  !$omp parallel
  do i=1,10
  end do
  !$omp end parallel

end subroutine

!CHECK-LABEL:  @_QPsb1
!CHECK:    %[[I_ADDR:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsb1Ei"}
!CHECK:    %[[I_DECL:.*]]:2 = hlfir.declare %[[I_ADDR]] {uniq_name = "_QFsb1Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    omp.parallel   {
!CHECK:      %[[I_PVT_ADDR:.*]] = fir.alloca i32 {bindc_name = "i", pinned}
!CHECK:      %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT_ADDR]] {uniq_name = "_QFsb1Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:      %[[I_FINAL_VAL:.*]]:2 = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[I_VAL:.*]] = %{{.*}}) -> (index, i32) {
!CHECK:        fir.store %[[I_VAL]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
!CHECK:      }
!CHECK:      fir.store %[[I_FINAL_VAL]]#1 to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
!CHECK:      omp.terminator
!CHECK:    }
!CHECK:    return
!CHECK:  }

subroutine sb2
  integer i, j, k
  !$omp parallel
  do j=1,10
    if (k .eq. 1) then
      do i=20, 30
      end do
    endif

    do i=40,50
    end do
  end do
  !$omp end parallel
end subroutine
!CHECK-LABEL:  @_QPsb2
!CHECK:    %[[I_ADDR:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsb2Ei"}
!CHECK:    %[[I_DECL:.*]]:2 = hlfir.declare %[[I_ADDR]] {uniq_name = "_QFsb2Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[J_ADDR:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFsb2Ej"}
!CHECK:    %[[J_DECL:.*]]:2 = hlfir.declare %[[J_ADDR]] {uniq_name = "_QFsb2Ej"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[K_ADDR:.*]] = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFsb2Ek"}
!CHECK:    %[[K_DECL:.*]]:2 = hlfir.declare %[[K_ADDR]] {uniq_name = "_QFsb2Ek"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    omp.parallel   {
!CHECK:      %[[I_PVT_ADDR:.*]] = fir.alloca i32 {bindc_name = "i", pinned}
!CHECK:      %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT_ADDR]] {uniq_name = "_QFsb2Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:      %[[J_PVT_ADDR:.*]] = fir.alloca i32 {bindc_name = "j", pinned}
!CHECK:      %[[J_PVT_DECL:.*]]:2 = hlfir.declare %[[J_PVT_ADDR]] {uniq_name = "_QFsb2Ej"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:      %[[FINAL_J_VAL:.*]]:2 = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[J_VAL:.*]] = %{{.*}}) -> (index, i32) {
!CHECK:        fir.store %arg1 to %9#1 : !fir.ref<i32>
!CHECK:        fir.if %{{.*}} {
!CHECK:          %[[FINAL_I_VAL:.*]]:2 = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[I_VAL:.*]] = %{{.*}}) -> (index, i32) {
!CHECK:            fir.store %[[I_VAL]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
!CHECK:          }
!CHECK:          fir.store %[[FINAL_I_VAL]]#1 to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
!CHECK:        }
!CHECK:        %[[FINAL_I_VAL:.*]]:2 = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[I_VAL:.*]] = %{{.*}}) -> (index, i32) {
!CHECK:          fir.store %[[I_VAL]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
!CHECK:        }
!CHECK:        fir.store %[[FINAL_I_VAL]]#1 to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
!CHECK:      }
!CHECK:      fir.store %[[FINAL_J_VAL]]#1 to %[[J_PVT_DECL]]#1 : !fir.ref<i32>
!CHECK:      omp.terminator
!CHECK:    }
!CHECK:    return
!CHECK:  }
