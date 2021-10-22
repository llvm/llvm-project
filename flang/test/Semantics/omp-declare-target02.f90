! RUN: %python %S/test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.1
! Check OpenMP construct validity for the following directives:
! 2.14.7 Declare Target Directive

program declare_target02
  integer :: arr1(10), arr1_to(10), arr1_link(10)
  common /blk1/ a1, a1_to, a1_link
  real, save :: eq_a, eq_b, eq_c, eq_d


  !$omp declare target (arr1)

  !$omp declare target (blk1)

  !ERROR: A variable in a DECLARE TARGET directive cannot be an element of a common block
  !$omp declare target (a1)

  !$omp declare target to (arr1_to)

  !$omp declare target to (blk1_to)

  !ERROR: A variable in a DECLARE TARGET directive cannot be an element of a common block
  !$omp declare target to (a1_to)

  !$omp declare target link (arr1_link)

  !$omp declare target link (blk1_link)

  !ERROR: A variable in a DECLARE TARGET directive cannot be an element of a common block
  !$omp declare target link (a1_link)

  equivalence(eq_a, eq_b)
  !ERROR: A variable in a DECLARE TARGET directive cannot appear in an EQUIVALENCE statement
  !$omp declare target (eq_a)

  !ERROR: A variable in a DECLARE TARGET directive cannot appear in an EQUIVALENCE statement
  !$omp declare target to (eq_a)

  !ERROR: A variable in a DECLARE TARGET directive cannot appear in an EQUIVALENCE statement
  !$omp declare target link (eq_b)

  !ERROR: A variable in a DECLARE TARGET directive cannot appear in an EQUIVALENCE statement
  !$omp declare target (eq_c)

  !ERROR: A variable in a DECLARE TARGET directive cannot appear in an EQUIVALENCE statement
  !$omp declare target to (eq_c)

  !ERROR: A variable in a DECLARE TARGET directive cannot appear in an EQUIVALENCE statement
  !$omp declare target link (eq_d)
  equivalence(eq_c, eq_d)

contains
  subroutine func()
    integer :: arr2(10), arr2_to(10), arr2_link(10)
    integer, save :: arr3(10), arr3_to(10), arr3_link(10)
    common /blk2/ a2, a2_to, a2_link
    common /blk3/ a3, a3_to, a3_link
    save /blk3/

    !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !$omp declare target (arr2)

    !$omp declare target (arr3)

    !ERROR: Implicitly typed local entity 'blk2' not allowed in specification expression
    !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !$omp declare target (blk2)

    !ERROR: A variable in a DECLARE TARGET directive cannot be an element of a common block
    !$omp declare target (a2)

    !ERROR: Implicitly typed local entity 'blk3' not allowed in specification expression
    !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !$omp declare target (blk3)

    !ERROR: A variable in a DECLARE TARGET directive cannot be an element of a common block
    !$omp declare target (a3)

    !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !$omp declare target to (arr2_to)

    !$omp declare target to (arr3_to)

    !ERROR: Implicitly typed local entity 'blk2_to' not allowed in specification expression
    !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !$omp declare target to (blk2_to)

    !ERROR: A variable in a DECLARE TARGET directive cannot be an element of a common block
    !$omp declare target to (a2_to)

    !ERROR: Implicitly typed local entity 'blk3_to' not allowed in specification expression
    !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !$omp declare target to (blk3_to)

    !ERROR: A variable in a DECLARE TARGET directive cannot be an element of a common block
    !$omp declare target to (a3_to)

    !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !$omp declare target link (arr2_link)

    !$omp declare target link (arr3_link)

    !ERROR: Implicitly typed local entity 'blk2_link' not allowed in specification expression
    !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !$omp declare target link (blk2_link)

    !ERROR: A variable in a DECLARE TARGET directive cannot be an element of a common block
    !$omp declare target link (a2_link)

    !ERROR: Implicitly typed local entity 'blk3_link' not allowed in specification expression
    !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !$omp declare target link (blk3_link)

    !ERROR: A variable in a DECLARE TARGET directive cannot be an element of a common block
    !$omp declare target link (a3_link)
  end
end

module mod4
  integer :: arr4(10), arr4_to(10), arr4_link(10)
  common /blk4/ a4, a4_to, a4_link

  !$omp declare target (arr4)

  !$omp declare target (blk4)

  !ERROR: A variable in a DECLARE TARGET directive cannot be an element of a common block
  !$omp declare target (a4)

  !$omp declare target to (arr4_to)

  !$omp declare target to (blk4_to)

  !ERROR: A variable in a DECLARE TARGET directive cannot be an element of a common block
  !$omp declare target to (a4_to)

  !$omp declare target link (arr4_link)

  !$omp declare target link (blk4_link)

  !ERROR: A variable in a DECLARE TARGET directive cannot be an element of a common block
  !$omp declare target link (a4_link)
end

subroutine func5()
  integer :: arr5(10), arr5_to(10), arr5_link(10)
  common /blk5/ a5, a5_to, a5_link

  !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
  !$omp declare target (arr5)

  !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
  !$omp declare target (blk5)

  !ERROR: A variable in a DECLARE TARGET directive cannot be an element of a common block
  !$omp declare target (a5)

  !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
  !$omp declare target to (arr5_to)

  !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
  !$omp declare target to (blk5_to)

  !ERROR: A variable in a DECLARE TARGET directive cannot be an element of a common block
  !$omp declare target to (a5_to)

  !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
  !$omp declare target link (arr5_link)

  !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
  !$omp declare target link (blk5_link)

  !ERROR: A variable in a DECLARE TARGET directive cannot be an element of a common block
  !$omp declare target link (a5_link)
end
