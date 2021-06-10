! RUN: %S/test_modfile.sh %s %t %flang_fc1
! REQUIRES: shell
! Check modfile generation for external interface
module m
  interface
    integer function f(x)
    end function
    subroutine s(y, z)
      logical y
      complex z
    end subroutine
  end interface
end

!Expect: m.mod
!module m
! interface
!  function f(x)
!   real(4)::x
!   integer(4)::f
!  end
! end interface
! interface
!  subroutine s(y,z)
!   logical(4)::y
!   complex(4)::z
!  end
! end interface
!end
