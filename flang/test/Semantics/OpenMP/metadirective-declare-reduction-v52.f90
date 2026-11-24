!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

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
  !$omp& otherwise(
!ERROR: Invalid reduction intrinsic operator in DECLARE REDUCTION directive
  !$omp& declare reduction(/ : integer : omp_out = omp_out + omp_in))
end

subroutine f03
  type :: t
    integer :: m
  end type
  !$omp metadirective otherwise(nothing)
  !$omp begin metadirective otherwise(parallel)
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
  !$omp& otherwise(error at(compilation))
end

subroutine f05
!WARNING: The declarative form of the METADIRECTIVE directive is not implemented yet and will be ignored [-Wopenmp-usage]
  !$omp metadirective &
  !$omp& when (implementation={vendor(llvm)}: &
!ERROR: Invalid reduction intrinsic operator in DECLARE REDUCTION directive
  !$omp& declare reduction(/ : integer : omp_out = omp_out + omp_in)) &
  !$omp& otherwise(error)
end

subroutine f06
!WARNING: The declarative form of the METADIRECTIVE directive is not implemented yet and will be ignored [-Wopenmp-usage]
  !$omp metadirective &
  !$omp& otherwise( &
!ERROR: Invalid reduction intrinsic operator in DECLARE REDUCTION directive
  !$omp& declare reduction(/ : integer : omp_out = omp_out + omp_in))
end

subroutine f07
  type :: t
    integer :: m
  end type
!WARNING: The declarative form of the METADIRECTIVE directive is not implemented yet and will be ignored [-Wopenmp-usage]
  !$omp metadirective &
  !$omp& otherwise( &
! Ok
  !$omp& declare reduction(+ : t : omp_out%m = omp_out%m + omp_in%m))
end
