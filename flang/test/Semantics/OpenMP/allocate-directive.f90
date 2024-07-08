! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags
! Check OpenMP Allocate directive
use omp_lib

! 2.11.3 declarative allocate
! 2.11.3 executable allocate

integer :: x, y
integer, allocatable :: a, b, m, n, t, z
!$omp allocate(x, y)
!$omp allocate(x, y) allocator(omp_default_mem_alloc)

!$omp allocate(a, b)
    allocate ( a, b )

!$omp allocate(a, b) allocator(omp_default_mem_alloc)
    allocate ( a, b )

!$omp allocate(t) allocator(omp_const_mem_alloc)
!$omp allocate(z) allocator(omp_default_mem_alloc)
!$omp allocate(m) allocator(omp_default_mem_alloc)
!$omp allocate(n)
    allocate ( t, z, m, n )

end
