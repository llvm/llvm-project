! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52
! OpenMP Version 5.0
! 2.4 Requires directive
! Target-related clauses in 'requires' directives must come strictly before any
! device constructs, such as declare target with device_type=nohost|any.

subroutine f
  integer, save :: x
  !WARNING: The usage of TO clause on DECLARE TARGET directive has been deprecated. Use ENTER clause instead.
  !$omp declare target to(x) device_type(nohost)
  !$omp declare target enter(x) device_type(nohost)
end subroutine f

subroutine g
  !ERROR: REQUIRES directive with 'DYNAMIC_ALLOCATORS' clause found lexically after device construct
  !$omp requires dynamic_allocators
  !ERROR: REQUIRES directive with 'REVERSE_OFFLOAD' clause found lexically after device construct
  !$omp requires reverse_offload
  !ERROR: REQUIRES directive with 'UNIFIED_ADDRESS' clause found lexically after device construct
  !$omp requires unified_address
  !ERROR: REQUIRES directive with 'UNIFIED_SHARED_MEMORY' clause found lexically after device construct
  !$omp requires unified_shared_memory
end subroutine g
