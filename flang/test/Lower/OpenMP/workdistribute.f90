! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtarget_teams_workdistribute
subroutine target_teams_workdistribute()
  ! CHECK: omp.target
  ! CHECK: omp.teams
  ! CHECK: omp.workdistribute
  !$omp target teams workdistribute
  ! CHECK: fir.call
  call f1()
  ! CHECK: omp.terminator
  ! CHECK: omp.terminator
  ! CHECK: omp.terminator
  !$omp end target teams workdistribute
end subroutine target_teams_workdistribute

! CHECK-LABEL: func @_QPteams_workdistribute
subroutine teams_workdistribute()
  ! CHECK: omp.teams
  ! CHECK: omp.workdistribute
  !$omp teams workdistribute
  ! CHECK: fir.call
  call f1()
  ! CHECK: omp.terminator
  ! CHECK: omp.terminator
  !$omp end teams workdistribute
end subroutine teams_workdistribute

! CHECK-LABEL: func @_QPtarget_teams_workdistribute_m
subroutine target_teams_workdistribute_m()
  ! CHECK: omp.target
  ! CHECK: omp.teams
  ! CHECK: omp.workdistribute
  !$omp target
  !$omp teams
  !$omp workdistribute
  ! CHECK: fir.call
  call f1()
  ! CHECK: omp.terminator
  ! CHECK: omp.terminator
  ! CHECK: omp.terminator
  !$omp end workdistribute
  !$omp end teams
  !$omp end target
end subroutine target_teams_workdistribute_m

! CHECK-LABEL: func @_QPteams_workdistribute_m
subroutine teams_workdistribute_m()
  ! CHECK: omp.teams
  ! CHECK: omp.workdistribute
  !$omp teams
  !$omp workdistribute
  ! CHECK: fir.call
  call f1()
  ! CHECK: omp.terminator
  ! CHECK: omp.terminator
  !$omp end workdistribute
  !$omp end teams
end subroutine teams_workdistribute_m
