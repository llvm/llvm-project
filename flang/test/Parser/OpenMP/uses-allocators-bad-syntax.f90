! REQUIRES: openmp_runtime

! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=52
! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=60

! Forms that the USES_ALLOCATORS parser does not accept.
!
! The first construct is a valid clause carrying no error annotation. A
! compiler without USES_ALLOCATORS parser support does not parse it either, so
! this test cannot pass vacuously.

subroutine uses_allocators_bad_syntax
  use omp_lib
  integer(omp_allocator_handle_kind) :: a, b
  integer :: x

  ! Sentinel: the canonical OpenMP 5.2 form parses.
  !$omp target uses_allocators(a)
  x = 1
  !$omp end target

  ! The clause takes at least one allocator specification.
  !ERROR: expected '='
  !$omp target uses_allocators()
  x = 2
  !$omp end target

  ! The OpenMP 6.0 semicolon-separated form for more than one
  ! clause-argument-specification is not implemented.
  !ERROR: expected ':'
  !$omp target uses_allocators(a; b)
  x = 3
  !$omp end target
end subroutine
