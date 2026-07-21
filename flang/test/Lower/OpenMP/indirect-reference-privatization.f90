! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - 2>&1 | FileCheck %s

!CHECK-LABEL: func @_QPparallel_simd
!CHECK: omp.parallel private(@_QFparallel_simdEk2_private_i32 {{.*}} -> %[[ARG:.*]] : !fir.ref<i32>)
!CHECK:   %[[PRIV_K2:.*]]:2 = hlfir.declare %[[ARG]] {uniq_name = "_QFparallel_simdEk2"}
!CHECK:   omp.simd linear(%[[PRIV_K2]]#0 {{.*}})

subroutine parallel_simd
  integer :: k1, k2
  !$omp parallel default(none)
    !$omp do
    do k1 = 1, 10
      do k2 = 1, 10
      end do
    end do
    !$omp end do

    !$omp simd linear(k2)
    do k2 = 1, 10
    end do
    !$omp end simd
  !$omp end parallel
end subroutine parallel_simd

!CHECK-LABEL: func @_QPtask_simd
!CHECK: omp.task private(@_QFtask_simdEk_firstprivate_i32 %{{.*}})
!CHECK:   %[[PRIV_K:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtask_simdEk"}
!CHECK:   omp.simd linear(%[[PRIV_K]]#0 : !fir.ref<i32> {{.*}})

subroutine task_simd
  integer :: k
  !$omp task
    !$omp simd linear(k)
      do k = 1, 10
      end do
    !$omp end simd
  !$omp end task
end subroutine task_simd
