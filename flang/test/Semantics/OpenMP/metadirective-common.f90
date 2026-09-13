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

subroutine f02_zero_score
  !$omp metadirective &
  !$omp & when(user={condition(score(0): .true.)}: nothing)
end

! Competing candidates reach scoring before the SCORE diagnostic is emitted.
! A score of -1 must not wrap the initial score to zero and crash selection.
subroutine f02_ranked_negative_scores(flag)
  logical :: flag
  integer :: i
  !$omp metadirective &
!ERROR: SCORE expression must be a non-negative constant integer expression
  !$omp & when(user={condition(score(-1): .true.)}: nothing) &
  !$omp & when(implementation={vendor(llvm)}: simd) otherwise(nothing)
  do i = 1, 4
  end do

  !$omp metadirective &
!ERROR: SCORE expression must be a non-negative constant integer expression
  !$omp & when(user={condition(score(-1): flag)}: nothing) &
  !$omp & when(implementation={vendor(llvm)}: simd) otherwise(nothing)
  do i = 1, 4
  end do

  !$omp metadirective &
!ERROR: SCORE expression must be a non-negative constant integer expression
  !$omp & when(implementation={vendor(score(-1): llvm)}: nothing) &
  !$omp & when(user={condition(.true.)}: simd) otherwise(nothing)
  do i = 1, 4
  end do
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

subroutine f05
  !$omp metadirective &
!ERROR: 'context-selector' modifier is required
  !$omp & when(nothing)
end
