! RUN: %flang_fc1 -emit-fir -O1 -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -O0 -fopenmp -fopenmp-version=60 %s -o - \
! RUN:   | FileCheck %s --check-prefix=CHECK-O0

! CHECK-LABEL: func @_QPtarget_teams_workdistribute
subroutine target_teams_workdistribute()
  use iso_fortran_env
  real(kind=real32) :: a
  real(kind=real32), dimension(10) :: x
  real(kind=real32), dimension(10) :: y
  !$omp target teams workdistribute

  ! CHECK: omp.target_data
  ! CHECK: omp.target
  ! CHECK: omp.teams
  ! CHECK: omp.parallel
  ! CHECK: omp.distribute
  ! CHECK: omp.wsloop
  ! CHECK: omp.loop_nest

  y = a * x + y

  ! CHECK: omp.target
  ! CHECK: omp.teams
  ! CHECK: omp.parallel
  ! CHECK: omp.distribute
  ! CHECK-NOT: fir.call @_FortranAAssign
  ! CHECK: omp.wsloop
  ! CHECK: omp.loop_nest

  y = 2.0_real32

  !$omp end target teams workdistribute
end subroutine target_teams_workdistribute

! CHECK-LABEL: func @_QPteams_workdistribute
subroutine teams_workdistribute()
  use iso_fortran_env
  real(kind=real32) :: a
  real(kind=real32), dimension(10) :: x
  real(kind=real32), dimension(10) :: y
  !$omp teams workdistribute

  ! CHECK: omp.teams
  ! CHECK: omp.parallel
  ! CHECK: omp.distribute
  ! CHECK: omp.wsloop
  ! CHECK: omp.loop_nest

  y = a * x + y

  ! CHECK: omp.teams
  ! CHECK: omp.parallel
  ! CHECK: omp.distribute
  ! CHECK-NOT: fir.call @_FortranAAssign
  ! CHECK: omp.wsloop
  ! CHECK: omp.loop_nest
  y = 2.0_real32

  !$omp end teams workdistribute
end subroutine teams_workdistribute

! At -O0, target teams workdistribute expands both assignments. On the host,
! the saxpy result write-back still uses the runtime, but the scalar broadcast
! is inlined and workshared.
! CHECK-O0-LABEL: func @_QPtarget_teams_workdistribute
! CHECK-O0-NOT: fir.call @_FortranAAssign

! CHECK-O0-LABEL: func @_QPteams_workdistribute
! CHECK-O0: omp.wsloop
! CHECK-O0: fir.call @_FortranAAssignSimple({{.*}}%c50_i32)
! CHECK-O0-NOT: fir.call @_FortranAAssign
! CHECK-O0: omp.wsloop
! CHECK-O0-NOT: fir.call @_FortranAAssign
