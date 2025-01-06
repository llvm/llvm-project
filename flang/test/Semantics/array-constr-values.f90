! RUN: %python %S/test_errors.py %s %flang_fc1
! Confirm enforcement of constraints and restrictions in 7.8
! C7110, C7111, C7112, C7113, C7114, C7115

subroutine arrayconstructorvalues()
  integer :: intarray(4)
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
  !ERROR: Value in array constructor of type 'CHARACTER(KIND=1,LEN=22_8)' could not be converted to the type of the array 'INTEGER(4)'
  intarray = [integer:: "RAM stores information", 2, 3, 4, 5]
  !ERROR: Value in array constructor of type 'employee' could not be converted to the type of the array 'INTEGER(4)'
  intarray = [integer:: EMPLOYEE (19, "Jack"), 2, 3, 4, 5]

  ! C7112
  !ERROR: Dimension 1 of left-hand side has extent 3, but right-hand side has extent 2
  !ERROR: Value in array constructor of type 'INTEGER(4)' could not be converted to the type of the array 'employee'
  emparray = (/ EMPLOYEE:: EMPLOYEE(19, "Ganesh"), EMPLOYEE(22, "Omkar"), 19 /)
  !ERROR: Dimension 1 of left-hand side has extent 3, but right-hand side has extent 2
  !ERROR: Value in array constructor of type 'employeer' could not be converted to the type of the array 'employee'
  emparray = (/ EMPLOYEE:: EMPLOYEE(19, "Ganesh"), EMPLOYEE(22, "Ram"),EMPLOYEER("ShriniwasPvtLtd") /)

  ! C7113
  !ERROR: Cannot have an unlimited polymorphic value in an array constructor
  intarray = (/ unlim_polymorphic, 2, 3, 4, 5/)

  ! C7114, F'2023 C7125
  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types INTEGER(4) and TYPE(base_type)
  !ERROR: ABSTRACT derived type 'base_type' may not be used in a structure constructor
  !ERROR: An item whose declared type is ABSTRACT may not appear in an array constructor
  !ERROR: Values in array constructor must have the same declared type when no explicit type appears
  intarray = (/ base_type(10), 2, 3, 4, 5 /)

  !ERROR: Item is not suitable for use in an array constructor
  intarray(1:1) = [ arrayconstructorvalues ]
end subroutine arrayconstructorvalues
subroutine checkC7115()
  real, dimension(10), parameter :: good1 = [(99.9, i = 1, 10)]
  real, dimension(100), parameter :: good2 = [((88.8, i = 1, 10), j = 1, 10)]
  real, dimension(-1:0), parameter :: good3 = [77.7, 66.6]
  !ERROR: Implied DO index 'i' is active in a surrounding implied DO loop and may not have the same name
  real, dimension(100), parameter :: bad = [((88.8, i = 1, 10), i = 1, 10)]

  !ERROR: Value of named constant 'bad2' ([INTEGER(4)::(int(j,kind=4),INTEGER(8)::j=1_8,1_8,0_8)]) cannot be computed as a constant value
  !ERROR: The stride of an implied DO loop must not be zero
  integer, parameter :: bad2(*) = [(j, j=1,1,0)]
  integer, parameter, dimension(-1:0) :: negLower = (/343,512/)
  integer, parameter, dimension(-1:0) :: negLower1 = ((/343,512/))

  real :: local

  local = good3(0)
  !ERROR: Subscript value (2) is out of range on dimension 1 in reference to a constant array value
  local = good3(2)
  call inner(negLower(:)) ! OK
  call inner(negLower1(:)) ! OK

  contains
    subroutine inner(arg)
      integer :: arg(:)
    end subroutine inner
end subroutine checkC7115
subroutine checkOkDuplicates
  real :: realArray(21) = &
    [ ((1.0, iDuplicate = 1,j), &
       (0.0, iDuplicate = j,3 ), &
        j = 1,5 ) ]
end subroutine
subroutine charLengths(c, array)
  character(3) :: c
  character(3) :: array(2)
  !No error should ensue for distinct but compatible DynamicTypes
  array = ["abc", c]
end subroutine
