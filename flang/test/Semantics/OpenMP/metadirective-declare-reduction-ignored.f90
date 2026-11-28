!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60 -Werror -Wno-experimental-option

! Check that we don't process the DECLARE_REDUCTION inside the METADIRECTIVE.
! The way that this test detects it is that it tries to trigger a conflict
! between a variable FOO and the reduction named FOO.

! Declarative METADIRECTIVEs are not handled yet. Once that's implemented
! this test will have to go away. At the moment it's just the check that
! unhandled situations don't interfere with valid code.

subroutine f00
  integer :: foo
  !WARNING: The declarative form of the METADIRECTIVE directive is not implemented yet and will be ignored [-Wopenmp-usage]
  !$omp metadirective when(user={condition(.false.)}: &
  !$omp &  declare_reduction(foo : integer) combiner(omp_out = omp_in))
end
