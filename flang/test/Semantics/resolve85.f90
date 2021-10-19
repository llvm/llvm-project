! RUN: %python %S/test_errors.py %s %flang_fc1
module m
! C730 The same type-attr-spec shall not appear more than once in a given 
! derived-type-stmt.
!
! R727 derived-type-stmt ->
!        TYPE [[, type-attr-spec-list] ::] type-name [( type-param-name-list )]
!  type-attr-spec values are:
!    ABSTRACT, PUBLIC, PRIVATE, BIND(C), EXTENDS(parent-type-name)
  !WARNING: Attribute 'ABSTRACT' cannot be used more than once
  type, abstract, public, abstract :: derived1
  end type derived1

  !WARNING: Attribute 'PUBLIC' cannot be used more than once
  type, public, abstract, public :: derived2
  end type derived2

  !WARNING: Attribute 'PRIVATE' cannot be used more than once
  type, private, abstract, private :: derived3
  end type derived3

  !ERROR: Attributes 'PUBLIC' and 'PRIVATE' conflict with each other
  type, public, abstract, private :: derived4
  end type derived4

  !WARNING: Attribute 'BIND(C)' cannot be used more than once
  type, bind(c), public, bind(c) :: derived5
  end type derived5

  type, public :: derived6
  end type derived6

  !ERROR: Attribute 'EXTENDS' cannot be used more than once
  type, extends(derived6), public, extends(derived6) :: derived7
  end type derived7

end module m
