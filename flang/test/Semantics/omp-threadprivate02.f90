! RUN: %python %S/test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.1
! Check OpenMP construct validity for the following directives:
! 2.21.2 Threadprivate Directive

program threadprivate02
  integer :: arr1(10)
  common /blk1/ a1
  real, save :: eq_a, eq_b, eq_c, eq_d

  !$omp threadprivate(arr1)

  !$omp threadprivate(/blk1/)

  !$omp threadprivate(blk1)

  !ERROR: A variable in a THREADPRIVATE directive cannot be an element of a common block
  !$omp threadprivate(a1)

  equivalence(eq_a, eq_b)
  !ERROR: A variable in a THREADPRIVATE directive cannot appear in an EQUIVALENCE statement
  !$omp threadprivate(eq_a)

  !ERROR: A variable in a THREADPRIVATE directive cannot appear in an EQUIVALENCE statement
  !$omp threadprivate(eq_c)
  equivalence(eq_c, eq_d)

contains
  subroutine func()
    integer :: arr2(10)
    integer, save :: arr3(10)
    common /blk2/ a2
    common /blk3/ a3
    save /blk3/

    !ERROR: A variable that appears in a THREADPRIVATE directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !$omp threadprivate(arr2)

    !$omp threadprivate(arr3)

    !$omp threadprivate(/blk2/)

    !ERROR: Implicitly typed local entity 'blk2' not allowed in specification expression
    !ERROR: A variable that appears in a THREADPRIVATE directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !$omp threadprivate(blk2)

    !ERROR: A variable in a THREADPRIVATE directive cannot be an element of a common block
    !$omp threadprivate(a2)

    !$omp threadprivate(/blk3/)

    !ERROR: Implicitly typed local entity 'blk3' not allowed in specification expression
    !ERROR: A variable that appears in a THREADPRIVATE directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !$omp threadprivate(blk3)

    !ERROR: A variable in a THREADPRIVATE directive cannot be an element of a common block
    !$omp threadprivate(a3)
  end
end

module mod4
  integer :: arr4(10)
  common /blk4/ a4

  !$omp threadprivate(arr4)

  !$omp threadprivate(/blk4/)

  !$omp threadprivate(blk4)

  !ERROR: A variable in a THREADPRIVATE directive cannot be an element of a common block
  !$omp threadprivate(a4)
end

subroutine func5()
  integer :: arr5(10)
  common /blk5/ a5

  !ERROR: A variable that appears in a THREADPRIVATE directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
  !$omp threadprivate(arr5)

  !$omp threadprivate(/blk5/)

  !ERROR: A variable that appears in a THREADPRIVATE directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
  !$omp threadprivate(blk5)

  !ERROR: A variable in a THREADPRIVATE directive cannot be an element of a common block
  !$omp threadprivate(a5)
end
