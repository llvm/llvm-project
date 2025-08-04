! RUN: %flang_fc1 -fdebug-unparse-no-sema -fopenmp %s | FileCheck %s
! Check Unparsing of OpenMP Allocate directive

program allocate_unparse
use omp_lib

real, dimension (:,:), allocatable :: darray
integer :: a, b, j, m, n, t, x, y, z

! 2.11.3 declarative allocate

!$omp allocate(x, y)
!$omp allocate(x, y) allocator(omp_default_mem_alloc)

! 2.11.3 executable allocate

!$omp allocate(a, b)
    allocate ( darray(a, b) )
!$omp allocate allocator(omp_default_mem_alloc)
    allocate ( darray(a, b) )
!$omp allocate(a, b) allocator(omp_default_mem_alloc)
    allocate ( darray(a, b) )

!$omp allocate(t) allocator(omp_const_mem_alloc)
!$omp allocate(z) allocator(omp_default_mem_alloc)
!$omp allocate(m) allocator(omp_default_mem_alloc)
!$omp allocate(n)
!$omp allocate(j) align(16)
    allocate ( darray(z, t) )

end program allocate_unparse

!CHECK:!$OMP ALLOCATE (x,y)
!CHECK:!$OMP ALLOCATE (x,y) ALLOCATOR(omp_default_mem_alloc)
!CHECK:!$OMP ALLOCATE (a,b)
!CHECK:ALLOCATE(darray(a,b))
!CHECK:!$OMP ALLOCATE ALLOCATOR(omp_default_mem_alloc)
!CHECK:ALLOCATE(darray(a,b))
!CHECK:!$OMP ALLOCATE (a,b) ALLOCATOR(omp_default_mem_alloc)
!CHECK:ALLOCATE(darray(a,b))
!CHECK:!$OMP ALLOCATE (t) ALLOCATOR(omp_const_mem_alloc)
!CHECK:!$OMP ALLOCATE (z) ALLOCATOR(omp_default_mem_alloc)
!CHECK:!$OMP ALLOCATE (m) ALLOCATOR(omp_default_mem_alloc)
!CHECK:!$OMP ALLOCATE (n)
!CHECK:!$OMP ALLOCATE (j) ALIGN(16)
!CHECK:ALLOCATE(darray(z,t))
