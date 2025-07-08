! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -Werror -fopenmp-version=52

! Check for deprecation of master directive and its combined/composite variants

subroutine test_master()
  integer :: c = 1
!WARNING: OpenMP directive MASTER has been deprecated, please use MASKED instead. [-Wopen-mp-usage]
  !$omp master 
  c = c + 1
  !$omp end master
end subroutine

subroutine test_parallel_master
  integer :: c = 2
!WARNING: OpenMP directive PARALLEL MASTER has been deprecated, please use PARALLEL MASKED instead. [-Wopen-mp-usage]
  !$omp parallel master
  c = c + 2
  !$omp end parallel master
end subroutine

subroutine test_master_taskloop_simd()
  integer :: i, j = 1
!WARNING: OpenMP directive MASTER TASKLOOP SIMD has been deprecated, please use MASKED TASKLOOP SIMD instead. [-Wopen-mp-usage]
  !$omp master taskloop simd 
  do i=1,10
   j = j + 1
  end do
  !$omp end master taskloop simd
end subroutine

subroutine test_master_taskloop
  integer :: i, j = 1
!WARNING: OpenMP directive MASTER TASKLOOP has been deprecated, please use MASKED TASKLOOP instead. [-Wopen-mp-usage]
  !$omp master taskloop
  do i=1,10
   j = j + 1
  end do
  !$omp end master taskloop 
end subroutine

subroutine test_parallel_master_taskloop_simd
  integer :: i, j = 1
!WARNING: OpenMP directive PARALLEL MASTER TASKLOOP SIMD has been deprecated, please use PARALLEL_MASKED TASKLOOP SIMD instead. [-Wopen-mp-usage]
  !$omp parallel master taskloop simd 
  do i=1,10
   j = j + 1
  end do
  !$omp end parallel master taskloop simd
end subroutine

subroutine test_parallel_master_taskloop
  integer :: i, j = 1
!WARNING: OpenMP directive PARALLEL MASTER TASKLOOP has been deprecated, please use PARALLEL MASKED TASKLOOP instead. [-Wopen-mp-usage]
  !$omp parallel master taskloop
  do i=1,10
   j = j + 1
  end do
  !$omp end parallel master taskloop 
end subroutine
