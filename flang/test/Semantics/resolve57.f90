! RUN: %S/test_errors.sh %s %t %f18
! Tests for the last sentence of C1128:
!A variable-name that is not permitted to appear in a variable definition
!context shall not appear in a LOCAL or LOCAL_INIT locality-spec.

subroutine s1(arg)
  real, intent(in) :: arg

  ! This is not OK because "arg" is "intent(in)"
!ERROR: INTENT IN argument 'arg' not allowed in a locality-spec
  do concurrent (i=1:5) local(arg)
  end do
end subroutine s1

subroutine s2(arg)
  real, value, intent(in) :: arg

  ! This is not OK even though "arg" has the "value" attribute.  C1128
  ! explicitly excludes dummy arguments of INTENT(IN)
!ERROR: INTENT IN argument 'arg' not allowed in a locality-spec
  do concurrent (i=1:5) local(arg)
  end do
end subroutine s2

module m3
  real, protected :: prot
  real var

  contains
    subroutine sub()
      ! C857 This is OK because of the "protected" attribute only applies to
      ! accesses outside the module
      do concurrent (i=1:5) local(prot)
      end do
    end subroutine sub
endmodule m3

subroutine s4()
  use m3

  ! C857 This is not OK because of the "protected" attribute
!ERROR: 'prot' may not appear in a locality-spec because it is not definable
  do concurrent (i=1:5) local(prot)
  end do

  ! C857 This is OK because of there's no "protected" attribute
  do concurrent (i=1:5) local(var)
  end do
end subroutine s4

subroutine s5()
  real :: a, b, c, d, e

  associate (a => b + c, d => e)
    b = 3.0
    ! C1101 This is OK because 'd' is associated with a variable
    do concurrent (i=1:5) local(d)
    end do

    ! C1101 This is not OK because 'a' is not associated with a variable
!ERROR: 'a' may not appear in a locality-spec because it is not definable
    do concurrent (i=1:5) local(a)
    end do
  end associate
end subroutine s5

subroutine s6()
  type point
    real :: x, y
  end type point

  type, extends(point) :: color_point
    integer :: color
  end type color_point

  type(point), target :: c, d
  class(point), pointer :: p_or_c

  p_or_c => c
  select type ( a => p_or_c )
  type is ( point )
    ! C1158 This is OK because 'a' is associated with a variable
    do concurrent (i=1:5) local(a)
    end do
  end select

  select type ( a => func() )
  type is ( point )
    ! C1158 This is not OK because 'a' is not associated with a variable
!ERROR: 'a' may not appear in a locality-spec because it is not definable
    do concurrent (i=1:5) local(a)
    end do
  end select

  contains
    function func()
      class(point), pointer :: func
      func => c
    end function func
end subroutine s6

module m4
  real, protected :: prot
  real var
endmodule m4

pure subroutine s7()
  use m4

  ! C1594 This is not OK because we're in a PURE subroutine
!ERROR: 'var' may not appear in a locality-spec because it is not definable
  do concurrent (i=1:5) local(var)
  end do
end subroutine s7

subroutine s8()
  integer, parameter :: iconst = 343

!ERROR: 'iconst' may not appear in a locality-spec because it is not definable
  do concurrent (i=1:5) local(iconst)
  end do
end subroutine s8
