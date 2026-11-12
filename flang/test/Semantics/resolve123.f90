! RUN: %python %S/test_errors.py %s %flang_fc1
! Tests for F'2023 C1131:
! A variable-name that appears in a REDUCE locality-spec shall not have the
! ASYNCHRONOUS, INTENT (IN), OPTIONAL, or VOLATILE attribute, shall not be
! coindexed, and shall not be an assumed-size array. A variable-name that is not
! permitted to appear in a variable definition context shall not appear in a
! REDUCE locality-spec.

subroutine s1()
! Cannot have ASYNCHRONOUS variable in a REDUCE locality spec
  integer, asynchronous :: k
!ERROR: ASYNCHRONOUS variable 'k' not allowed in a REDUCE locality-spec
  do concurrent(i=1:5) reduce(+:k)
     k = k + i
  end do
end subroutine s1

subroutine s2(arg)
! Cannot have a dummy OPTIONAL in a REDUCE locality spec
  integer, optional :: arg
!ERROR: OPTIONAL argument 'arg' not allowed in a locality-spec
  do concurrent(i=1:5) reduce(*:arg)
     arg = arg * 1
  end do
end subroutine s2

subroutine s3(arg)
! This is OK
  real :: arg
  integer :: reduce, reduce2, reduce3
  do concurrent(i=1:5) reduce(max:arg,reduce) reduce(iand:reduce2,reduce3)
     arg = max(arg, i)
     reduce = max(reduce, i)
     reduce3 = iand(reduce3, i)
  end do
end subroutine s3

subroutine s4(arg)
! Cannot have a dummy INTENT(IN) in a REDUCE locality spec
  real, intent(in) :: arg
!ERROR: INTENT IN argument 'arg' not allowed in a locality-spec
  do concurrent(i=1:5) reduce(min:arg)
!ERROR: Left-hand side of assignment is not definable
!ERROR: 'arg' is an INTENT(IN) dummy argument
     arg = min(arg, i)
  end do
end subroutine s4

module m
contains
  subroutine s5()
    ! Cannot have VOLATILE variable in a REDUCE locality spec
    integer, volatile :: var
    !ERROR: VOLATILE variable 'var' not allowed in a REDUCE locality-spec
    do concurrent(i=1:5) reduce(ieor:var)
       var = ieor(var, i)
    end do
  end subroutine s5
  subroutine f(x)
    integer :: x
  end subroutine f
end module m

subroutine s8(arg)
! Cannot have an assumed size array
  integer, dimension(*) :: arg
!ERROR: Assumed size array 'arg' not allowed in a locality-spec
  do concurrent(i=1:5) reduce(ior:arg)
     arg(i) = ior(arg(i), i)
  end do
end subroutine s8

subroutine s9()
! Reduction variable should not appear in a variable definition context
  integer :: i
!ERROR: 'i' is already declared in this scoping unit
  do concurrent(i=1:5) reduce(+:i)
  end do
end subroutine s9

subroutine s10()
! Cannot have variable inside of a NAMELIST in a REDUCE locality spec
  integer :: k
  namelist /nlist1/ k
!ERROR: NAMELIST variable 'k' not allowed in a REDUCE locality-spec
  do concurrent(i=1:5) reduce(+:k)
     k = k + i
  end do
end subroutine s10
