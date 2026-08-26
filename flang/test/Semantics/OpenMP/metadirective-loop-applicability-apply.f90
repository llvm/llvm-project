!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

! A loop transformation in APPLY inherits the reachability of the
! metadirective replacement that contains it.

subroutine f01()
  !$omp metadirective &
  !$omp& when(user={condition(score(10): .true.)}: nothing) &
  !$omp& when(user={condition(score(5): .true.)}: &
  !$omp& tile sizes(2) apply(grid: unroll)) &
  !$omp& otherwise(nothing)
end subroutine

subroutine f02(flag)
  logical :: flag
  !$omp metadirective &
  !$omp& when(user={condition(score(10): flag)}: nothing) &
  !$omp& when(user={condition(score(5): .true.)}: &
  !ERROR: This construct should contain a DO-loop or a loop-nest-generating construct
  !ERROR: This construct should contain a DO-loop or a loop-nest-generating construct
  !$omp& tile sizes(2) apply(grid: unroll)) &
  !$omp& otherwise(nothing)
end subroutine
