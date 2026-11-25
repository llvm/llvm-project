! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52
! OpenMP Version 5.0
! 2.4 Requires directive
! Target-related clauses in 'requires' directives must come strictly before any
! device constructs in the same program unit, such as declare target with
! device_type=nohost|any.

subroutine f
  integer, save :: x
  !WARNING: TO clause is no longer allowed on DECLARE TARGET directive since OpenMP v5.2 [-Wopenmp-deprecated]
  !WARNING: The usage of TO clause on DECLARE TARGET directive has been deprecated. Use ENTER clause instead. [-Wopenmp-deprecated]
  !$omp declare target to(x) device_type(nohost)
  !$omp declare target enter(x) device_type(nohost)
  !ERROR: REQUIRES directive with 'DYNAMIC_ALLOCATORS' clause found lexically after device construct
  !$omp requires dynamic_allocators
  !WARNING: REVERSE_OFFLOAD clause is not supported and will be ignored
  !ERROR: REQUIRES directive with 'REVERSE_OFFLOAD' clause found lexically after device construct
  !$omp requires reverse_offload
  !ERROR: REQUIRES directive with 'UNIFIED_ADDRESS' clause found lexically after device construct
  !$omp requires unified_address
  !ERROR: REQUIRES directive with 'UNIFIED_SHARED_MEMORY' clause found lexically after device construct
  !$omp requires unified_shared_memory
end subroutine f