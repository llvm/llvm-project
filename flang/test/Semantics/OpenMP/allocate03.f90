! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags
! OpenMP Version 5.0
! 2.11.3 allocate Directive
! A variable that is part of another variable (as an array or
! structure element) cannot appear in an allocate directive.
subroutine allocate()
use omp_lib

  type my_type
    integer :: array(10)
  end type my_type
  type(my_type) :: my_var
  real, dimension (:,:), allocatable :: darray
  integer :: a, b

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the ALLOCATE directive
  !$omp allocate(my_var%array)

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the ALLOCATE directive
  !$omp allocate(darray, my_var%array) allocator(omp_default_mem_alloc)
    allocate ( darray(a, b) )

end subroutine allocate
