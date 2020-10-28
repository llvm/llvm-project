! RUN: %S/test_errors.sh %s %t %f18
! Confirm enforcement of constraints and restrictions in 7.8
! C7110, C7111, C7112, C7113, C7114, C7115

subroutine arrayconstructorvalues()
  integer :: intarray(5)
  integer(KIND=8) :: k8 = 20

  TYPE EMPLOYEE
    INTEGER AGE
    CHARACTER (LEN = 30) NAME
  END TYPE EMPLOYEE
  TYPE EMPLOYEER
    CHARACTER (LEN = 30) NAME
  END TYPE EMPLOYEER

  TYPE(EMPLOYEE) :: emparray(3)
  class(*), pointer :: unlim_polymorphic
  TYPE, ABSTRACT :: base_type
    INTEGER :: CARPRIZE
  END TYPE
  ! Different declared type
  !ERROR: Values in array constructor must have the same declared type when no explicit type appears
  intarray = (/ 1, 2, 3, 4., 5/)  ! C7110
  ! Different kind type parameter
  !ERROR: Values in array constructor must have the same declared type when no explicit type appears
  intarray = (/ 1,2,3,4, k8 /)    ! C7110

  ! C7111
  !ERROR: Value in array constructor of type 'LOGICAL(4)' could not be converted to the type of the array 'INTEGER(4)'
  intarray = [integer:: .true., 2, 3, 4, 5]
  !ERROR: Value in array constructor of type 'CHARACTER(1)' could not be converted to the type of the array 'INTEGER(4)'
  intarray = [integer:: "RAM stores information", 2, 3, 4, 5]
  !ERROR: Value in array constructor of type 'employee' could not be converted to the type of the array 'INTEGER(4)'
  intarray = [integer:: EMPLOYEE (19, "Jack"), 2, 3, 4, 5]

  ! C7112
  !ERROR: Value in array constructor of type 'INTEGER(4)' could not be converted to the type of the array 'employee'
  emparray = (/ EMPLOYEE:: EMPLOYEE(19, "Ganesh"), EMPLOYEE(22, "Omkar"), 19 /)
  !ERROR: Value in array constructor of type 'employeer' could not be converted to the type of the array 'employee'
  emparray = (/ EMPLOYEE:: EMPLOYEE(19, "Ganesh"), EMPLOYEE(22, "Ram"),EMPLOYEER("ShriniwasPvtLtd") /)

  ! C7113
  !ERROR: Cannot have an unlimited polymorphic value in an array constructor
  !ERROR: Values in array constructor must have the same declared type when no explicit type appears
  intarray = (/ unlim_polymorphic, 2, 3, 4, 5/)

  ! C7114
  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types INTEGER(4) and TYPE(base_type)
  !ERROR: ABSTRACT derived type 'base_type' may not be used in a structure constructor
  !ERROR: Values in array constructor must have the same declared type when no explicit type appears
  intarray = (/ base_type(10), 2, 3, 4, 5 /)
end subroutine arrayconstructorvalues
subroutine checkC7115()
  real, dimension(10), parameter :: good1 = [(99.9, i = 1, 10)]
  real, dimension(100), parameter :: good2 = [((88.8, i = 1, 10), j = 1, 10)]
  !ERROR: Implied DO index is active in surrounding implied DO loop and may not have the same name
  !ERROR: 'i' is already declared in this scoping unit
  real, dimension(100), parameter :: bad = [((88.8, i = 1, 10), i = 1, 10)]
end subroutine checkC7115
