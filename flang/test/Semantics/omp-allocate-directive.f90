! RUN: %S/test_errors.sh %s %t %flang_fc1 -fopenmp
! REQUIRES: shell
! Check OpenMP Allocate directive
use omp_lib

! 2.11.3 declarative allocate
! 2.11.3 executable allocate

real, dimension (:,:), allocatable :: darray
integer :: a, b, x, y, m, n, t, z
!$omp allocate(x, y)
!$omp allocate(x, y) allocator(omp_default_mem_alloc)

!$omp allocate(a, b)
    allocate ( darray(a, b) )

!$omp allocate(a, b) allocator(omp_default_mem_alloc)
    allocate ( darray(a, b) )

!$omp allocate(t) allocator(omp_const_mem_alloc)
!$omp allocate(z) allocator(omp_default_mem_alloc)
!$omp allocate(m) allocator(omp_default_mem_alloc)
!$omp allocate(n)
    allocate ( darray(z, t) )

end
