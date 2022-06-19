! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in ALLOCATE statements

! Creating a symbol that allocate should accept
module share
  real, pointer :: rp
end module share

module m
! Creating symbols that allocate should not accept
  type :: a_type
    real, allocatable :: x
    contains
      procedure, pass :: foo => mfoo
      procedure, pass :: bar => mbar
  end type

contains
  function mfoo(x)
    class(a_type) :: x
    class(a_type), allocatable :: foo
    foo = x
  end function
  subroutine mbar(x)
    class(a_type) :: x
  end subroutine
end module

subroutine C932(ed1, ed5, ed7, edc9, edc10, okad1, okpd1, okacd5)
! Each allocate-object shall be a data pointer or an allocatable variable.
  use :: share
  use :: m, only: a_type
  type TestType1
    integer, allocatable :: ok(:)
    integer :: nok(10)
  end type
  type TestType2
    integer, pointer :: ok
    integer :: nok
  end type
  interface
    function foo(x)
      real(4) :: foo, x
    end function
    subroutine bar()
    end subroutine
  end interface
  real ed1(:), e2
  real, save :: e3[*]
  real , target :: e4, ed5(:)
  real , parameter :: e6 = 5.
  procedure(foo), pointer :: proc_ptr1 => NULL()
  procedure(bar), pointer :: proc_ptr2
  type(TestType1) ed7
  type(TestType2) e8
  type(TestType1) edc9[*]
  type(TestType2) edc10[*]
  class(a_type), allocatable :: a_var

  real, allocatable :: oka1(:, :), okad1(:, :), oka2
  real, pointer :: okp1(:, :), okpd1(:, :), okp2
  real, pointer, save :: okp3
  real, allocatable, save :: oka3, okac4[:,:]
  real, allocatable :: okacd5(:, :)[:]

  !ERROR: Name in ALLOCATE statement must be a variable name
  allocate(foo)
  !ERROR: Name in ALLOCATE statement must be a variable name
  allocate(bar)
  !ERROR: Name in ALLOCATE statement must be a variable name
  allocate(C932)
  !ERROR: Name in ALLOCATE statement must be a variable name
  allocate(proc_ptr1)
  !ERROR: Name in ALLOCATE statement must be a variable name
  allocate(proc_ptr2)
  !ERROR: Name in ALLOCATE statement must be a variable name
  allocate(a_var%foo)
  !ERROR: Name in ALLOCATE statement must be a variable name
  allocate(a_var%bar)

  !ERROR: Entity in ALLOCATE statement must have the ALLOCATABLE or POINTER attribute
  allocate(ed1)
  !ERROR: Entity in ALLOCATE statement must have the ALLOCATABLE or POINTER attribute
  allocate(e2)
  !ERROR: Entity in ALLOCATE statement must have the ALLOCATABLE or POINTER attribute
  allocate(e3)
  !ERROR: Entity in ALLOCATE statement must have the ALLOCATABLE or POINTER attribute
  allocate(e4)
  !ERROR: Entity in ALLOCATE statement must have the ALLOCATABLE or POINTER attribute
  allocate(ed5)
  !ERROR: Name in ALLOCATE statement must be a variable name
  allocate(e6)
  !ERROR: Entity in ALLOCATE statement must have the ALLOCATABLE or POINTER attribute
  allocate(ed7)
  !ERROR: Entity in ALLOCATE statement must have the ALLOCATABLE or POINTER attribute
  allocate(ed7%nok(2))
  !ERROR: Entity in ALLOCATE statement must have the ALLOCATABLE or POINTER attribute
  allocate(ed8)
  !ERROR: Entity in ALLOCATE statement must have the ALLOCATABLE or POINTER attribute
  allocate(ed8)
  !ERROR: Entity in ALLOCATE statement must have the ALLOCATABLE or POINTER attribute
  allocate(edc9%nok)
  !ERROR: Entity in ALLOCATE statement must have the ALLOCATABLE or POINTER attribute
  allocate(edc10)

  ! No errors expected below:
  allocate(a_var)
  allocate(a_var%x)
  allocate(oka1(5, 7), okad1(4, 8), oka2)
  allocate(okp1(5, 7), okpd1(4, 8), okp2)
  allocate(okp1(5, 7), okpd1(4, 8), okp2)
  allocate(okp3, oka3)
  allocate(okac4[2:4,4:*])
  allocate(okacd5(1:2,3:4)[5:*])
  allocate(ed7%ok(7))
  allocate(e8%ok)
  allocate(edc9%ok(4))
  allocate(edc10%ok)
  allocate(rp)
end subroutine
