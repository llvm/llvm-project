! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for C1801 - C1805

module m
  public s
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

  ! ERROR: A derived type with the BIND attribute cannot have a type bound procedure
  type, bind(c) :: t4
    integer :: x
   contains
    procedure, nopass :: b => s
  end type

  ! WARNING: A derived type with the BIND attribute is empty
  type, bind(c) :: t5
  end type

  ! ERROR: A derived type with the BIND attribute cannot have a pointer or allocatable component
  type, bind(c) :: t6
    integer, pointer :: x
  end type

  ! ERROR: A derived type with the BIND attribute cannot have a pointer or allocatable component
  type, bind(c) :: t7
    integer, allocatable :: y
  end type

  ! ERROR: The component of the interoperable derived type must have the BIND attribute
  type :: t8
    integer :: x
  end type

  type, bind(c) :: t9
    type(t8) :: y
    integer :: z
  end type

end
