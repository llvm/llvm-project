! RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=61 %s -o - | FileCheck %s

!===============================================================================
! `num_threads` clause with dims modifier (OpenMP 6.1)
!===============================================================================

! CHECK-LABEL: func @_QPparallel_numthreads_dims4
subroutine parallel_numthreads_dims4()
  ! CHECK: omp.parallel
  ! CHECK-SAME: num_threads(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i32, i32, i32, i32)
  !$omp parallel num_threads(dims(4): 4, 5, 6, 7)
  call f1()
  ! CHECK: omp.terminator
  !$omp end parallel
end subroutine parallel_numthreads_dims4

! CHECK-LABEL: func @_QPparallel_numthreads_dims2
subroutine parallel_numthreads_dims2()
  ! CHECK: omp.parallel
  ! CHECK-SAME: num_threads(%{{.*}}, %{{.*}} : i32, i32)
  !$omp parallel num_threads(dims(2): 8, 4)
  call f1()
  ! CHECK: omp.terminator
  !$omp end parallel
end subroutine parallel_numthreads_dims2

! CHECK-LABEL: func @_QPparallel_numthreads_dims_var
subroutine parallel_numthreads_dims_var(a, b, c)
  integer, intent(in) :: a, b, c
  ! CHECK: omp.parallel
  ! CHECK-SAME: num_threads(%{{.*}}, %{{.*}}, %{{.*}} : i32, i32, i32)
  !$omp parallel num_threads(dims(3): a, b, c)
  call f1()
  ! CHECK: omp.terminator
  !$omp end parallel
end subroutine parallel_numthreads_dims_var

!===============================================================================
! `num_threads` clause without dims modifier (legacy)
!===============================================================================

! CHECK-LABEL: func @_QPparallel_numthreads_legacy
subroutine parallel_numthreads_legacy(n)
  integer, intent(in) :: n
  ! CHECK: omp.parallel
  ! CHECK-SAME: num_threads(%{{.*}} : i32)
  !$omp parallel num_threads(n)
  call f1()
  ! CHECK: omp.terminator
  !$omp end parallel
end subroutine parallel_numthreads_legacy

! CHECK-LABEL: func @_QPparallel_numthreads_const
subroutine parallel_numthreads_const()
  ! CHECK: omp.parallel
  ! CHECK-SAME: num_threads(%{{.*}} : i32)
  !$omp parallel num_threads(16)
  call f1()
  ! CHECK: omp.terminator
  !$omp end parallel
end subroutine parallel_numthreads_const
