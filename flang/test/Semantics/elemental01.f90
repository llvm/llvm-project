! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Tests ELEMENTAL subprogram constraints C15100-15102

!ERROR: An ELEMENTAL subroutine may not have an alternate return dummy argument
elemental subroutine altret(*)
end subroutine

elemental subroutine arrarg(a)
  !ERROR: A dummy argument of an ELEMENTAL procedure must be scalar
  real, intent(in) :: a(1)
end subroutine

elemental subroutine alloarg(a)
  !ERROR: A dummy argument of an ELEMENTAL procedure may not be ALLOCATABLE
  real, intent(in), allocatable :: a
end subroutine

elemental subroutine coarg(a)
  !ERROR: A dummy argument of an ELEMENTAL procedure may not be a coarray
  real, intent(in) :: a[*]
end subroutine

elemental subroutine ptrarg(a)
  !ERROR: A dummy argument of an ELEMENTAL procedure may not be a POINTER
  real, intent(in), pointer :: a
end subroutine

impure elemental subroutine barearg(a)
  !ERROR: A dummy argument of an ELEMENTAL procedure must have an INTENT() or VALUE attribute
  real :: a
end subroutine

elemental function arrf(n)
  integer, value :: n
  !ERROR: The result of an ELEMENTAL function must be scalar
  real :: arrf(n)
end function

elemental function allof(n)
  integer, value :: n
  !ERROR: The result of an ELEMENTAL function may not be ALLOCATABLE
  real, allocatable :: allof
end function

elemental function ptrf(n)
  integer, value :: n
  !ERROR: The result of an ELEMENTAL function may not be a POINTER
  real, pointer :: ptrf
end function

module m
  integer modvar
  type t
    character(:), allocatable :: c
  end type
  type pdt(L)
    integer, len :: L
  end type
  type container
    class(pdt(:)), allocatable :: c
  end type
 contains
  !ERROR: Invalid specification expression for elemental function result: dependence on value of dummy argument 'n'
  elemental character(n) function bad1(n)
    integer, intent(in) :: n
  end
  !ERROR: Invalid specification expression for elemental function result: non-constant inquiry function 'len' not allowed for local object
  elemental character(x%c%len) function bad2(x)
    type(t), intent(in) :: x
  end
  !ERROR: Invalid specification expression for elemental function result: non-constant type parameter inquiry not allowed for local object
  elemental character(x%c%L) function bad3(x)
    class(container), intent(in) :: x
  end
  elemental character(len(x)) function ok1(x) ! ok
    character(*), intent(in) :: x
  end
  elemental character(modvar) function ok2(x) ! ok
    character(*), intent(in) :: x
  end
  elemental character(len(x)) function ok3(x) ! ok
    character(modvar), intent(in) :: x
  end
  elemental character(storage_size(x)) function ok4(x) ! ok
    class(*), intent(in) :: x
  end
end
