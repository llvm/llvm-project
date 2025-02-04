! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50

! OpenMP Version 5.0
! Check OpenMP construct validity for the following directives:
! 11.7 Loop directive

program main
  integer :: i, x

  !$omp teams 
  !ERROR: `BIND(TEAMS)` must be specified since the `LOOP` region is strictly nested inside a `TEAMS` region.
  !$omp loop bind(thread)
  do i = 1, 10
    x = x + 1
  end do
  !$omp end loop
  !$omp end teams

  !ERROR: `BIND(TEAMS)` must be specified since the `LOOP` directive is combined with a `TEAMS` construct.
  !$omp target teams loop bind(thread)
  do i = 1, 10
    x = x + 1
  end do
  !$omp end target teams loop

  !ERROR: `BIND(TEAMS)` must be specified since the `LOOP` directive is combined with a `TEAMS` construct.
  !$omp teams loop bind(thread)
  do i = 1, 10
    x = x + 1
  end do
  !$omp end teams loop

end program main
