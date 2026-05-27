! RUN: %python %S/test_modfile.py %s %flang_fc1
! Check that recognized compiler directives in interface blocks
! survive module file serialization and parsing.

module m
  interface
    subroutine sub(x)
!dir$ ignore_tkr(t) x
      real, intent(in) :: x
    end subroutine
  end interface
end module

!Expect: m.mod
!module m
! interface
!  subroutine sub(x)
!   real(4),intent(in)::x
!   !dir$ ignore_tkr(t) x
!  end
! end interface
!end

module m2
  interface
    subroutine sub2(a)
!dir$ ignore_tkr(kr) a
      integer, intent(in) :: a
    end subroutine
  end interface
end module

!Expect: m2.mod
!module m2
! interface
!  subroutine sub2(a)
!   integer(4),intent(in)::a
!   !dir$ ignore_tkr(kr) a
!  end
! end interface
!end

