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
