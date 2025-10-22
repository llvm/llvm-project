! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=52
! OpenMP Version 5.2
! The allocate clause's allocator modifier must be of type allocator_handle
! and the align modifier must be constant, positive integer expression

program allocate_align_tree
    use omp_lib
    integer, allocatable :: j(:), xarray(:)
    integer :: z, t, xx
    t = 2
    z = 3
    !ERROR: Must be a constant value
!$omp allocate(j) align(xx)
    !WARNING: The executable form of the OpenMP ALLOCATE directive has been deprecated, please use ALLOCATORS instead [-Wopen-mp-usage]
    !ERROR: The alignment should be positive
!$omp allocate(xarray) align(-32) allocator(omp_large_cap_mem_alloc)
    allocate(j(z), xarray(t))
end program allocate_align_tree

