! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.0
! 2.11.3 allocate Directive 
! A type parameter inquiry cannot appear in an allocate directive.

subroutine allocate()
use omp_lib
  type my_type(kind_param, len_param)
    INTEGER, KIND :: kind_param
    INTEGER, LEN :: len_param
    INTEGER :: array(10)
  end type

  type(my_type(2, 4)) :: my_var
  INTEGER(KIND=4) :: x
  CHARACTER(LEN=32) :: w
  INTEGER, DIMENSION(:), ALLOCATABLE :: y
  
  !ERROR: A type parameter inquiry cannot appear on the ALLOCATE directive
  !$omp allocate(x%KIND)
  
  !ERROR: A type parameter inquiry cannot appear on the ALLOCATE directive
  !$omp allocate(w%LEN)

  !ERROR: A type parameter inquiry cannot appear on the ALLOCATE directive
  !$omp allocate(y%KIND)
  
  !ERROR: A type parameter inquiry cannot appear on the ALLOCATE directive
  !$omp allocate(my_var%kind_param)
 
  !ERROR: A type parameter inquiry cannot appear on the ALLOCATE directive
  !$omp allocate(my_var%len_param)

end subroutine allocate

