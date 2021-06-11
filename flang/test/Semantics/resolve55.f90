! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Tests for C1128:
! A variable-name that appears in a LOCAL or LOCAL_INIT locality-spec shall not
! have the ALLOCATABLE; INTENT (IN); or OPTIONAL attribute; shall not be of
! finalizable type; shall not be a nonpointer polymorphic dummy argument; and
! shall not be a coarray or an assumed-size array.

subroutine s1()
! Cannot have ALLOCATABLE variable in a locality spec
  integer, allocatable :: k
!ERROR: ALLOCATABLE variable 'k' not allowed in a locality-spec
  do concurrent(i=1:5) local(k)
  end do
end subroutine s1

subroutine s2(arg)
! Cannot have a dummy OPTIONAL in a locality spec
  integer, optional :: arg
!ERROR: OPTIONAL argument 'arg' not allowed in a locality-spec
  do concurrent(i=1:5) local(arg)
  end do
end subroutine s2

subroutine s3(arg)
! This is OK
  real :: arg
  do concurrent(i=1:5) local(arg)
  end do
end subroutine s3

subroutine s4(arg)
! Cannot have a dummy INTENT(IN) in a locality spec
  real, intent(in) :: arg
!ERROR: INTENT IN argument 'arg' not allowed in a locality-spec
  do concurrent(i=1:5) local(arg)
  end do
end subroutine s4

module m
! Cannot have a variable of a finalizable type in a locality spec
  type t1
    integer :: i
  contains
    final :: f
  end type t1
 contains
  subroutine s5()
    type(t1) :: var
    !ERROR: Finalizable variable 'var' not allowed in a locality-spec
    do concurrent(i=1:5) local(var)
    end do
  end subroutine s5
  subroutine f(x)
    type(t1) :: x
  end subroutine f
end module m

subroutine s6
! Cannot have a nonpointer polymorphic dummy argument in a locality spec
  type :: t
    integer :: field
  end type t
contains
  subroutine s(x, y)
    class(t), pointer :: x
    class(t) :: y

! This is allowed
    do concurrent(i=1:5) local(x)
    end do

! This is not allowed
!ERROR: Nonpointer polymorphic argument 'y' not allowed in a locality-spec
    do concurrent(i=1:5) local(y)
    end do
  end subroutine s
end subroutine s6

subroutine s7()
! Cannot have a coarray
  integer, codimension[*] :: coarray_var
!ERROR: Coarray 'coarray_var' not allowed in a locality-spec
  do concurrent(i=1:5) local(coarray_var)
  end do
end subroutine s7

subroutine s8(arg)
! Cannot have an assumed size array
  integer, dimension(*) :: arg
!ERROR: Assumed size array 'arg' not allowed in a locality-spec
  do concurrent(i=1:5) local(arg)
  end do
end subroutine s8
