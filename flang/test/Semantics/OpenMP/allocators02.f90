! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags
! OpenMP Version 5.2
! 6.7 allocators construct
! A variable that is part of another variable (as an array or
! structure element) cannot appear in an allocatprs construct.

subroutine allocate()
use omp_lib

  type my_type
    integer, allocatable :: array(:)
  end type my_type

  type(my_type) :: my_var

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear on the ALLOCATORS directive
  !$omp allocators allocate(my_var%array)
    allocate(my_var%array(10))

end subroutine allocate
