! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for C1801 - C1805

module m
  public s
  !ERROR: Interoperable array must have at least one element
  real, bind(c) :: x(0)
contains
  subroutine s
  end
end

program main
  use m
  type, abstract :: v
    integer :: i
  end type

  ! ERROR: A derived type with the BIND attribute cannot have the SEQUENCE attribute
  type, bind(c) :: t1
    sequence
    integer :: x
  end type

  ! ERROR: A derived type with the BIND attribute has type parameter(s)
  type, bind(c) :: t2(k)
    integer, KIND :: k
    integer :: x
  end type

  ! ERROR: A derived type with the BIND attribute cannot extend from another derived type
  type, bind(c), extends(v) :: t3
    integer :: x
  end type

  type, bind(c) :: t4
    integer :: x
   contains
    ! ERROR: A derived type with the BIND attribute cannot have a type bound procedure
    procedure, nopass :: b => s
  end type

  ! WARNING: A derived type with the BIND attribute is empty
  type, bind(c) :: t5
  end type

  type, bind(c) :: t6
    ! ERROR: A derived type with the BIND attribute cannot have a pointer or allocatable component
    integer, pointer :: x
  end type

  type, bind(c) :: t7
    ! ERROR: A derived type with the BIND attribute cannot have a pointer or allocatable component
    integer, allocatable :: y
  end type

  type :: t8
    integer :: x
  end type

  type, bind(c) :: t9
    !ERROR: Component 'y' of an interoperable derived type must have the BIND attribute
    type(t8) :: y
    integer :: z
  end type

  type, bind(c) :: t10
    !WARNING: A CHARACTER component of a BIND(C) type should have length 1
    character(len=2) x
  end type
  type, bind(c) :: t11
    !ERROR: Each component of an interoperable derived type must have an interoperable type
    character(kind=2) x
  end type
  type, bind(c) :: t12
    !PORTABILITY: A LOGICAL component of a BIND(C) type should have the interoperable KIND=C_BOOL
    logical(kind=8) x
  end type
  type, bind(c) :: t13
    !ERROR: Each component of an interoperable derived type must have an interoperable type
    real(kind=2) x
  end type
  type, bind(c) :: t14
    !ERROR: Each component of an interoperable derived type must have an interoperable type
    complex(kind=2) x
  end type
  type, bind(c) :: t15
    !ERROR: An array component of an interoperable type must have at least one element
    real :: x(0)
  end type

end
