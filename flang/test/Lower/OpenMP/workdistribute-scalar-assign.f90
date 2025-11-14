! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

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
  ! CHECK: fir.call @_FortranAAssign
  !$omp teams workdistribute
  aa = 20
  !$omp end teams workdistribute

end subroutine teams_workdistribute_scalar_assign
