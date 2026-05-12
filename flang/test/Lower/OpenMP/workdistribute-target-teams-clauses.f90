! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtarget_teams_workdistribute
! CHECK: omp.target_data map_entries({{.*}})
! CHECK: omp.target thread_limit({{.*}}) host_eval({{.*}}) map_entries({{.*}})
! CHECK: omp.teams num_teams({{.*}})
! CHECK: omp.parallel
! CHECK: omp.distribute
! CHECK: omp.wsloop
! CHECK: omp.loop_nest

subroutine target_teams_workdistribute()
  use iso_fortran_env
  real(kind=real32) :: a
  real(kind=real32), dimension(10) :: x
  real(kind=real32), dimension(10) :: y
  integer :: i

  a = 2.0_real32
  x = [(real(i, real32), i = 1, 10)]
  y = [(real(i * 0.5, real32), i = 1, 10)]

  !$omp target teams workdistribute &
  !$omp&  num_teams(4) &
  !$omp&  thread_limit(8) &
  !$omp&  default(shared) &
  !$omp&  private(i) &
  !$omp&  map(to: x) &
  !$omp&  map(tofrom: y)
  y = a * x + y
  !$omp end target teams workdistribute
end subroutine target_teams_workdistribute
