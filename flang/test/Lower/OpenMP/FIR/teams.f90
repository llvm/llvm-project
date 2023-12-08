! RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPteams_simple
subroutine teams_simple()
  ! CHECK: omp.teams
  !$omp teams
  ! CHECK: fir.call
  call f1()
  ! CHECK: omp.terminator
  !$omp end teams
end subroutine teams_simple

!===============================================================================
! `num_teams` clause
!===============================================================================

! CHECK-LABEL: func @_QPteams_numteams
subroutine teams_numteams(num_teams)
  integer, intent(inout) :: num_teams

  ! CHECK: omp.teams
  ! CHECK-SAME: num_teams( to %{{.*}}: i32)
  !$omp teams num_teams(4)
  ! CHECK: fir.call
  call f1()
  ! CHECK: omp.terminator
  !$omp end teams

  ! CHECK: omp.teams
  ! CHECK-SAME: num_teams( to %{{.*}}: i32)
  !$omp teams num_teams(num_teams)
  ! CHECK: fir.call
  call f2()
  ! CHECK: omp.terminator
  !$omp end teams

end subroutine teams_numteams

!===============================================================================
! `if` clause
!===============================================================================

! CHECK-LABEL: func @_QPteams_if
subroutine teams_if(alpha)
  integer, intent(in) :: alpha
  logical :: condition

  ! CHECK: omp.teams
  ! CHECK-SAME: if(%{{.*}})
  !$omp teams if(.false.)
  ! CHECK: fir.call
  call f1()
  ! CHECK: omp.terminator
  !$omp end teams

  ! CHECK: omp.teams
  ! CHECK-SAME: if(%{{.*}})
  !$omp teams if(alpha .le. 0)
  ! CHECK: fir.call
  call f2()
  ! CHECK: omp.terminator
  !$omp end teams

  ! CHECK: omp.teams
  ! CHECK-SAME: if(%{{.*}})
  !$omp teams if(condition)
  ! CHECK: fir.call
  call f3()
  ! CHECK: omp.terminator
  !$omp end teams
end subroutine teams_if

!===============================================================================
! `thread_limit` clause
!===============================================================================

! CHECK-LABEL: func @_QPteams_threadlimit
subroutine teams_threadlimit(thread_limit)
  integer, intent(inout) :: thread_limit

  ! CHECK: omp.teams
  ! CHECK-SAME: thread_limit(%{{.*}}: i32)
  !$omp teams thread_limit(4)
  ! CHECK: fir.call
  call f1()
  ! CHECK: omp.terminator
  !$omp end teams

  ! CHECK: omp.teams
  ! CHECK-SAME: thread_limit(%{{.*}}: i32)
  !$omp teams thread_limit(thread_limit)
  ! CHECK: fir.call
  call f2()
  ! CHECK: omp.terminator
  !$omp end teams

end subroutine teams_threadlimit

!===============================================================================
! `allocate` clause
!===============================================================================

! CHECK-LABEL: func @_QPteams_allocate
subroutine teams_allocate()
   use omp_lib
   integer :: x
   ! CHECK: omp.teams
   ! CHECK-SAME: allocate(%{{.+}} : i32 -> %{{.+}} : !fir.ref<i32>)
   !$omp teams allocate(omp_high_bw_mem_alloc: x) private(x)
   ! CHECK: arith.addi
   x = x + 12
   ! CHECK: omp.terminator
   !$omp end teams
end subroutine teams_allocate
