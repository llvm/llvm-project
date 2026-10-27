!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51

subroutine f01
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: &
!ERROR: Invalid reduction intrinsic operator in DECLARE REDUCTION directive
  !$omp& declare reduction(/ : integer : omp_out = omp_out + omp_in))
end

subroutine f02
  !$omp metadirective &
  !$omp& default(
!ERROR: Invalid reduction intrinsic operator in DECLARE REDUCTION directive
  !$omp& declare reduction(/ : integer : omp_out = omp_out + omp_in))
end

subroutine f03
  type :: t
    integer :: m
  end type
  !$omp metadirective default(nothing)
  !$omp begin metadirective default(parallel)
  block
! Ok
    !$omp declare reduction(+ : t : omp_out%m = omp_out%m + omp_in%m)
  end block
  !$omp end metadirective
end

subroutine f04
  !$omp metadirective &
  !$omp& when (implementation={vendor(llvm)}: &
!ERROR: Invalid reduction intrinsic operator in DECLARE REDUCTION directive
  !$omp& declare reduction(/ : integer : omp_out = omp_out + omp_in)) &
  !$omp& default(error at(compilation))
end

subroutine f05
  !$omp metadirective &
  !$omp& when (implementation={vendor(llvm)}: &
!ERROR: Invalid reduction intrinsic operator in DECLARE REDUCTION directive
  !$omp& declare reduction(/ : integer : omp_out = omp_out + omp_in)) &
  !$omp& default(error)
end

subroutine f06
  !$omp metadirective &
  !$omp& default( &
!ERROR: Invalid reduction intrinsic operator in DECLARE REDUCTION directive
  !$omp& declare reduction(/ : integer : omp_out = omp_out + omp_in))
end

subroutine f07
  !$omp metadirective &
  !$omp& default( &
! Ok
  !$omp& declare reduction(plus : integer : omp_out = omp_out + omp_in))
end
