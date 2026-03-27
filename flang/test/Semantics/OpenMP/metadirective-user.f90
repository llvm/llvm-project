!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

! The USER trait set

subroutine f00(x)
  integer :: x
  !$omp metadirective &
!ERROR: CONDITION trait requires a single LOGICAL expression
  !$omp & when(user={condition(score(2): x)}: nothing)
end

subroutine f01
  !$omp metadirective &
!ERROR: CONDITION trait requires a single expression property
  !$omp & when(user={condition(.true., .false.)}: nothing)
end

subroutine f02
  !$omp metadirective &
!ERROR: Extension traits are not valid for USER trait set
  !$omp & when(user={fred}: nothing)
end

subroutine f03(x)
  integer :: x
  !$omp metadirective &
!This is ok
  !$omp & when(user={condition(x > 0)}: nothing)
end
