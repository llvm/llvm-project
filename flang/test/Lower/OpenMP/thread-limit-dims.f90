! RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=61 %s -o - | FileCheck %s

!===============================================================================
! `thread_limit` clause with dims modifier (OpenMP 6.1)
!===============================================================================

! CHECK-LABEL: func @_QPteams_threadlimit_dims3
subroutine teams_threadlimit_dims3()
  ! CHECK: omp.teams
  ! CHECK-SAME: thread_limit(%{{.*}}, %{{.*}}, %{{.*}} : i32, i32, i32)
  !$omp teams thread_limit(dims(3): 16, 8, 4)
  call f1()
  ! CHECK: omp.terminator
  !$omp end teams
end subroutine teams_threadlimit_dims3

! CHECK-LABEL: func @_QPteams_threadlimit_dims2
subroutine teams_threadlimit_dims2()
  ! CHECK: omp.teams
  ! CHECK-SAME: thread_limit(%{{.*}}, %{{.*}} : i32, i32)
  !$omp teams thread_limit(dims(2): 32, 16)
  call f1()
  ! CHECK: omp.terminator
  !$omp end teams
end subroutine teams_threadlimit_dims2

! CHECK-LABEL: func @_QPteams_threadlimit_dims_var
subroutine teams_threadlimit_dims_var(a, b, c)
  integer, intent(in) :: a, b, c
  ! CHECK: omp.teams
  ! CHECK-SAME: thread_limit(%{{.*}}, %{{.*}}, %{{.*}} : i32, i32, i32)
  !$omp teams thread_limit(dims(3): a, b, c)
  call f1()
  ! CHECK: omp.terminator
  !$omp end teams
end subroutine teams_threadlimit_dims_var

!===============================================================================
! `thread_limit` clause without dims modifier (legacy)
!===============================================================================

! CHECK-LABEL: func @_QPteams_threadlimit_legacy
subroutine teams_threadlimit_legacy(n)
  integer, intent(in) :: n
  ! CHECK: omp.teams
  ! CHECK-SAME: thread_limit(%{{.*}} : i32)
  !$omp teams thread_limit(n)
  call f1()
  ! CHECK: omp.terminator
  !$omp end teams
end subroutine teams_threadlimit_legacy

! CHECK-LABEL: func @_QPteams_threadlimit_const
subroutine teams_threadlimit_const()
  ! CHECK: omp.teams
  ! CHECK-SAME: thread_limit(%{{.*}} : i32)
  !$omp teams thread_limit(64)
  call f1()
  ! CHECK: omp.terminator
  !$omp end teams
end subroutine teams_threadlimit_const
