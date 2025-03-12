!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

! Common context selector tests

subroutine f00
  !$omp metadirective &
  !$omp & when(implementation={vendor("this")}, &
!ERROR: Repeated trait set name IMPLEMENTATION in a context specifier
  !$omp &      implementation={requires(unified_shared_memory)}: nothing)
end

subroutine f01
  !$omp metadirective &
!ERROR: Repeated trait name ISA in a trait set
  !$omp & when(device={isa("this"), isa("that")}: nothing)
end

subroutine f02
  !$omp metadirective &
!ERROR: SCORE expression must be a non-negative constant integer expression
  !$omp & when(user={condition(score(-2): .true.)}: nothing)
end

subroutine f03(x)
  integer :: x
  !$omp metadirective &
!ERROR: SCORE expression must be a non-negative constant integer expression
  !$omp & when(user={condition(score(x): .true.)}: nothing)
end

subroutine f04
  !$omp metadirective &
!ERROR: Trait property should be a scalar expression
!ERROR: More invalid properties are present
  !$omp & when(target_device={device_num("device", "foo"(1))}: nothing)
end

