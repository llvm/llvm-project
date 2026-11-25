!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51

subroutine f01
!WARNING: The declarative form of the METADIRECTIVE directive is not implemented yet and will be ignored [-Wopenmp-usage]
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: &
!ERROR: Invalid reduction intrinsic operator in DECLARE REDUCTION directive
  !$omp& declare reduction(/ : integer : omp_out = omp_out + omp_in))
end

subroutine f02
!WARNING: The declarative form of the METADIRECTIVE directive is not implemented yet and will be ignored [-Wopenmp-usage]
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
!WARNING: The declarative form of the METADIRECTIVE directive is not implemented yet and will be ignored [-Wopenmp-usage]
  !$omp metadirective &
  !$omp& when (implementation={vendor(llvm)}: &
!ERROR: Invalid reduction intrinsic operator in DECLARE REDUCTION directive
  !$omp& declare reduction(/ : integer : omp_out = omp_out + omp_in)) &
  !$omp& default(error at(compilation))
end

subroutine f05
!WARNING: The declarative form of the METADIRECTIVE directive is not implemented yet and will be ignored [-Wopenmp-usage]
  !$omp metadirective &
  !$omp& when (implementation={vendor(llvm)}: &
!ERROR: Invalid reduction intrinsic operator in DECLARE REDUCTION directive
  !$omp& declare reduction(/ : integer : omp_out = omp_out + omp_in)) &
  !$omp& default(error)
end

subroutine f06
!WARNING: The declarative form of the METADIRECTIVE directive is not implemented yet and will be ignored [-Wopenmp-usage]
  !$omp metadirective &
  !$omp& default( &
!ERROR: Invalid reduction intrinsic operator in DECLARE REDUCTION directive
  !$omp& declare reduction(/ : integer : omp_out = omp_out + omp_in))
end

subroutine f07
  type :: t
    integer :: m
  end type
!WARNING: The declarative form of the METADIRECTIVE directive is not implemented yet and will be ignored [-Wopenmp-usage]
  !$omp metadirective &
  !$omp& default( &
! Ok
  !$omp& declare reduction(+ : t : omp_out%m = omp_out%m + omp_in%m))
end
