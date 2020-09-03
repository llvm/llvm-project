! RUN: %S/test_errors.sh %s %t %f18
! Test DO loop semantics for constraint C1130 --
! The constraint states that "If the locality-spec DEFAULT ( NONE ) appears in a
! DO CONCURRENT statement; a variable that is a local or construct entity of a
! scope containing the DO CONCURRENT construct; and that appears in the block of
! the construct; shall have its locality explicitly specified by that
! statement."

module m
  real :: mvar
end module m

subroutine s1()
  use m
  integer :: i, ivar, jvar, kvar
  real :: x

  type point
    real :: x, y
  end type point

  type, extends(point) :: color_point
    integer :: color
  end type color_point

  type(point), target :: c
  class(point), pointer :: p_or_c

  p_or_c => c

  jvar = 5
  
  ! References in this DO CONCURRENT are OK since there's no DEFAULT(NONE)
  ! locality-spec
  associate (avar => ivar)
    do concurrent (i = 1:2) shared(jvar)
      ivar = 3
      ivar = ivar + i
      block
        real :: bvar
        avar = 4
        x = 3.5
        bvar = 3.5 + i
      end block
      jvar = 5
      mvar = 3.5
    end do
  end associate
  
  associate (avar => ivar)
!ERROR: DO CONCURRENT step expression may not be zero
    do concurrent (i = 1:2:0) default(none) shared(jvar) local(kvar)
!ERROR: Variable 'ivar' from an enclosing scope referenced in DO CONCURRENT with DEFAULT(NONE) must appear in a locality-spec
      ivar =  &
!ERROR: Variable 'ivar' from an enclosing scope referenced in DO CONCURRENT with DEFAULT(NONE) must appear in a locality-spec
        ivar + i
      block
        real :: bvar
!ERROR: Variable 'avar' from an enclosing scope referenced in DO CONCURRENT with DEFAULT(NONE) must appear in a locality-spec
        avar = 4
!ERROR: Variable 'x' from an enclosing scope referenced in DO CONCURRENT with DEFAULT(NONE) must appear in a locality-spec
        x = 3.5
        bvar = 3.5 + i ! OK, bvar's scope is within the DO CONCURRENT
      end block
      jvar = 5 ! OK, jvar appears in a locality spec
      kvar = 5 ! OK, kvar appears in a locality spec

!ERROR: Variable 'mvar' from an enclosing scope referenced in DO CONCURRENT with DEFAULT(NONE) must appear in a locality-spec
      mvar = 3.5
    end do
  end associate

  select type ( a => p_or_c )
  type is ( point )
    do concurrent (i=1:5) local(a)
      ! C1130 This is OK because there's no DEFAULT(NONE) locality spec
      a%x = 3.5
    end do
  end select

  select type ( a => p_or_c )
  type is ( point )
    do concurrent (i=1:5) default (none)
!ERROR: Variable 'a' from an enclosing scope referenced in DO CONCURRENT with DEFAULT(NONE) must appear in a locality-spec
      a%x = 3.5
    end do
  end select

  select type ( a => p_or_c )
  type is ( point )
    do concurrent (i=1:5) default (none) local(a)
      ! C1130 This is OK because 'a' is in a locality-spec
      a%x = 3.5
    end do
  end select

  x = 5.0  ! OK, we're not in a DO CONCURRENT
  
end subroutine s1
