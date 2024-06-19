! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags
! OpenMP Version 5.0
! 2.11.3 allocate Directive
! List items specified in an allocate directive that is associated
! with an allocate statement must be variables that are allocated
! by the allocate statement.

subroutine allocate()
use omp_lib
  integer, dimension(:), allocatable :: a, b, c, d, e, f, &
                                        g, h, i, j, k, l

  !$omp allocate(a) allocator(omp_default_mem_alloc)
    allocate(a(1), b(2))

  !$omp allocate(c, d) allocator(omp_default_mem_alloc)
    allocate(c(3), d(4))

  !$omp allocate(e) allocator(omp_default_mem_alloc)
  !$omp allocate(f, g) allocator(omp_default_mem_alloc)
  !$omp allocate
    allocate(e(5), f(6), g(7))

  !ERROR: Object 'i' in ALLOCATE directive not found in corresponding ALLOCATE statement
  !$omp allocate(h, i) allocator(omp_default_mem_alloc)
    allocate(h(8))

  !ERROR: Object 'j' in ALLOCATE directive not found in corresponding ALLOCATE statement
  !$omp allocate(j, k) allocator(omp_default_mem_alloc)
  !$omp allocate(l) allocator(omp_default_mem_alloc)
    allocate(k(9), l(10))

end subroutine allocate
