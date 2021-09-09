! RUN: %python %S/test_errors.py %s %flang -fopenmp

! OpenMP Version 5.0
! Check OpenMP construct validity for the following directives:
! 2.7 Teams Construct

program main
  integer :: i, j, N = 10
  real :: a, b, c

  !$omp teams
  a = 3.14
  !$omp end teams

  !$omp target
  !$omp teams
  a = 3.14
  !$omp end teams
  !$omp end target

  !$omp target
  !$omp parallel
  !ERROR: TEAMS region can only be strictly nested within the implicit parallel region or TARGET region
  !$omp teams
  a = 3.14
  !$omp end teams
  !$omp end parallel
  !$omp end target

  !$omp parallel
  !ERROR: TEAMS region can only be strictly nested within the implicit parallel region or TARGET region
  !$omp teams
  a = 3.14
  !$omp end teams
  !$omp end parallel

  !$omp do
  do i = 1, N
  !ERROR: TEAMS region can only be strictly nested within the implicit parallel region or TARGET region
  !$omp teams
  a = 3.14
  !$omp end teams
  end do

  !$omp master
  !ERROR: TEAMS region can only be strictly nested within the implicit parallel region or TARGET region
  !$omp teams
  a = 3.14
  !$omp end teams
  !$omp end master

  !$omp target parallel
  !ERROR: TEAMS region can only be strictly nested within the implicit parallel region or TARGET region
  !$omp teams
  a = 3.14
  !$omp end teams
  !$omp end target parallel

  !$omp target
  !$omp teams
  !ERROR: Only `DISTRIBUTE` or `PARALLEL` regions are allowed to be strictly nested inside `TEAMS` region.
  !ERROR: TEAMS region can only be strictly nested within the implicit parallel region or TARGET region
  !$omp teams
  a = 3.14
  !$omp end teams
  !$omp end teams
  !$omp end target

  !$omp target teams
  !ERROR: TEAMS region can only be strictly nested within the implicit parallel region or TARGET region
  !$omp teams
  a = 3.14
  !$omp end teams
  !$omp end target teams

  !ERROR: TARGET construct with nested TEAMS region contains statements or directives outside of the TEAMS construct
  !$omp target
  do i = 1, N
    !$omp teams
    a = 3.14
    !$omp end teams
  enddo
  !$omp end target

  !ERROR: TARGET construct with nested TEAMS region contains statements or directives outside of the TEAMS construct
  !$omp target
  if (i .GT. 1) then
    if (j .GT. 1) then
      !$omp teams
      a = 3.14
      !$omp end teams
    end if
  end if
  !$omp end target

  !ERROR: TARGET construct with nested TEAMS region contains statements or directives outside of the TEAMS construct
  !$omp target
  b = 3.14
  !$omp teams
  a = 3.14
  !$omp end teams
  !$omp end target

  !ERROR: TARGET construct with nested TEAMS region contains statements or directives outside of the TEAMS construct
  !$omp target
  !$omp teams
  a = 3.14
  !$omp end teams
  c = 3.14
  !$omp end target

end program main
