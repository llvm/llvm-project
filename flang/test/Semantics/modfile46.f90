! RUN: %python %S/test_modfile.py %s %flang_fc1
! Ensure that interfaces, which are internal to procedures and are used to
! define the interface of dummy or return value procedures, are included in
! .mod files.
module m
  implicit none
contains
  function f(x)
    real, intent(in) :: x
    abstract interface
       subroutine used_int(x, p)
         implicit none
         real, intent(out) :: x
         interface
            subroutine inner_int(x)
              implicit none
              real, intent(out) :: x
            end subroutine inner_int
         end interface
         procedure(inner_int) :: p
       end subroutine used_int

       pure logical function unused_int(i)
         implicit none
         integer, intent(in) :: i
       end function unused_int
    end interface
    procedure(used_int), pointer :: f

    f => null()
  contains
    subroutine internal()
    end subroutine internal
  end function f
end module m

!Expect: m.mod
!module m
!contains
!function f(x)
!real(4),intent(in)::x
!procedure(used_int),pointer::f
!abstract interface
!subroutine used_int(x,p)
!real(4),intent(out)::x
!procedure(inner_int)::p
!interface
!subroutine inner_int(x)
!real(4),intent(out)::x
!end
!end interface
!end
!end interface
!end
!end
