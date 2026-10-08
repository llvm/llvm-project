! RUN: %flang_fc1 -emit-fir -O1 -fopenmp -fopenmp-version=60 %s -o - \
! RUN:   | FileCheck %s --implicit-check-not="fir.call @_FortranAAssign"
! RUN: %flang_fc1 -emit-fir -O0 -fopenmp -fopenmp-version=60 %s -o - \
! RUN:   | FileCheck %s --check-prefix=CHECK-O0

! CHECK-LABEL: func @_QPtarget_teams_workdistribute_scalar_assign
subroutine target_teams_workdistribute_scalar_assign()
  integer :: aa(10)

  ! CHECK: omp.target_data
  ! CHECK: omp.target
  ! CHECK: omp.teams
  ! CHECK: omp.parallel
  ! CHECK: omp.distribute
  ! CHECK: omp.wsloop
  ! CHECK: omp.loop_nest

  !$omp target teams workdistribute
  aa = 20
  !$omp end target teams workdistribute

end subroutine target_teams_workdistribute_scalar_assign

! CHECK-LABEL: func @_QPteams_workdistribute_scalar_assign
subroutine teams_workdistribute_scalar_assign()
  integer :: aa(10)
  ! CHECK: omp.teams
  ! CHECK: omp.parallel
  ! CHECK: omp.distribute
  ! CHECK: omp.wsloop
  ! CHECK: omp.loop_nest
  !$omp teams workdistribute
  aa = 20
  !$omp end teams workdistribute

end subroutine teams_workdistribute_scalar_assign

! At -O0 host (no -fopenmp-is-target-device), target teams workdistribute
! still goes through workdistributeRuntimeCallLower, so the scalar
! broadcast inside the target region is inlined and workshared, with no
! _FortranAAssign runtime call.
! CHECK-O0-LABEL: func @_QPtarget_teams_workdistribute_scalar_assign
! CHECK-O0: omp.wsloop
! CHECK-O0-NOT: fir.call @_FortranAAssign

! At -O0 host, teams workdistribute (no target) does not inline the
! scalar broadcast after PR #201774, so the assignment remains a
! _FortranAAssign runtime call. This pins the plain host -O0 behavior.
! CHECK-O0-LABEL: func @_QPteams_workdistribute_scalar_assign
! CHECK-O0: fir.call @_FortranAAssign
