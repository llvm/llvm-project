! RUN: %python %S/test_errors.py %s %flang_fc1
! C735 If EXTENDS appears, SEQUENCE shall not appear.
! C738 The same private-or-sequence shall not appear more than once in a
! given derived-type-def .
!
! C740 If SEQUENCE appears,
!  the type shall have at least one component,
!  each data component shall be declared to be of an intrinsic type or of a sequence type,
!  the derived type shall not have any type parameter,
!  and a type-bound-procedure-part shall not appear.
subroutine s1
  integer :: t0
  !ERROR: 't0' is not a derived type
  type(t0) :: x
  type :: t1
  end type
  type, extends(t1) :: t2
  end type
  !ERROR: Derived type 't3' not found
  type, extends(t3) :: t4
  end type
  !ERROR: 't0' is not a derived type
  type, extends(t0) :: t5
  end type
end subroutine

module m1
  type t0
  end type
end
module m2
  type t
  end type
end
module m3
  type t0
  end type
end
subroutine s2
  use m1
  use m2, t0 => t
  use m3
  !ERROR: Reference to 't0' is ambiguous
  type, extends(t0) :: t1
  end type
end subroutine

module m4
  type :: t1
    private
    sequence
    private  ! not a fatal error
    sequence ! not a fatal error
    real :: t1Field
  end type
  type :: t1a
  end type
  !ERROR: A sequence type may not have the EXTENDS attribute
  type, extends(t1a) :: t2
    sequence
    integer i
  end type
  type :: t3
    sequence
    integer i
  !ERROR: A sequence type may not have a CONTAINS statement
  contains
  end type
  !ERROR: A sequence type must have at least one component
  type :: emptyType
    sequence
  end type emptyType
  type :: plainType
    real :: plainField
  end type plainType
  type :: sequenceType
    sequence
    real :: sequenceField
  end type sequenceType
  type :: testType
    sequence
    !ERROR: A sequence type data component must either be of an intrinsic type or a derived sequence type
    class(*), allocatable :: typeStarField
    !ERROR: A sequence type data component must either be of an intrinsic type or a derived sequence type
    type(plainType) :: testField1
    !Pointers are ok as an extension
    type(plainType), pointer :: testField1p
    type(sequenceType) :: testField2
    procedure(real), pointer, nopass :: procField
  end type testType
  !ERROR: A sequence type may not have type parameters
  type :: paramType(param)
    integer, kind :: param
    sequence
    real :: paramField
  end type paramType
contains
  subroutine s3
    type :: t1
      !ERROR: PRIVATE is only allowed in a derived type that is in a module
      private
    contains
      !ERROR: PRIVATE is only allowed in a derived type that is in a module
      private
    end type
  end
end
