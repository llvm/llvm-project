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

subroutine f04(a, b)
  logical :: a, b
  !$omp metadirective &
!ERROR: Repeated trait name CONDITION in a trait set
  !$omp & when(user={condition(a), condition(b)}: nothing)
end

! Profiling still runs after expression diagnostics have been recorded.
subroutine invalid_negative_kind(n)
  integer :: n
!ERROR: CONDITION trait requires a single LOGICAL expression
!ERROR: INTEGER(KIND=3) is not a supported type
  !$omp metadirective when(user={condition(n == -1_3)}: taskyield) otherwise(taskwait)
end

subroutine invalid_complex_integer_kind()
!ERROR: CONDITION trait requires a single LOGICAL expression
!ERROR: INTEGER(KIND=3) is not a supported type
  !$omp metadirective when(user={condition((1_3, 2) == (1, 2))}: taskyield) otherwise(taskwait)
end

subroutine invalid_complex_real_kind()
!ERROR: CONDITION trait requires a single LOGICAL expression
!ERROR: Unsupported REAL(KIND=7)
  !$omp metadirective when(user={condition((1.0_7, 0.0) == (1.0, 0.0))}: taskyield) otherwise(taskwait)
end
