! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.1
! Check OpenMP construct validity for the following directives:
! 2.21.2 Threadprivate Directive

program threadprivate02
  integer :: arr1(10)
  common /blk1/ a1
  real, save :: eq_a, eq_b, eq_c, eq_d
  integer :: eq_e, eq_f
  equivalence(eq_e, eq_f)
  common /blk2/ eq_e

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

  ! This is an extension to the OpenMP semantics, see https://github.com/llvm/llvm-project/issues/180493
  !WARNING: A variable in a THREADPRIVATE directive used in an EQUIVALENCE statement is an OpenMP extension (variable 'eq_e' from common block '/blk2/') [-Wopenmp-threadprivate-equivalence]
  !$omp threadprivate(/blk2/)

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

    !ERROR: A variable in a THREADPRIVATE directive cannot be an element of a common block
    !$omp threadprivate(a2)

    !$omp threadprivate(/blk3/)

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

subroutine func6
    common /foo/ l
    integer :: l
    integer :: k(l)
    save
    !ERROR: An automatic data object cannot appear in THREADPRIVATE, because it cannot be given the SAVE attribute
    !$omp threadprivate(k)
end subroutine func6

subroutine func7()
    integer :: x
    common /blk/ x
    save
    !ERROR: A variable in a THREADPRIVATE directive cannot be an element of a common block
    !$omp threadprivate(x)

    ! PASS
    !$omp threadprivate(/blk/)
end subroutine

module mod_func8
    integer, save :: x
contains
    subroutine func8()
        !ERROR: The THREADPRIVATE directive and the common block or variable in it must appear in the same declaration section of a scoping unit
        !$omp threadprivate(x)
    end subroutine
end module

subroutine func9()
    use mod_func8
    !ERROR: The THREADPRIVATE directive and the common block or variable in it must appear in the same declaration section of a scoping unit
    !$omp threadprivate(x)
end subroutine

subroutine func10()
    !ERROR: A variable that appears in a THREADPRIVATE directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !$omp threadprivate(x)
    x = 1
end subroutine

subroutine func11
    type :: t
      integer :: a = 12
    end type t
    type(t) :: x
    integer :: i, j

    save :: x, j
    !ERROR: A variable that appears in a THREADPRIVATE directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !$omp threadprivate(x, i, j)
end subroutine

subroutine func12(a)
    integer :: a(:)
    save
    !ERROR: A dummy argument cannot appear in THREADPRIVATE, because it cannot be given the SAVE attribute
    !$omp threadprivate(a)
end subroutine

integer function func13()
    save
    !ERROR: A function result object cannot appear in THREADPRIVATE, because it cannot be given the SAVE attribute
    !$omp threadprivate(func13)
end function
