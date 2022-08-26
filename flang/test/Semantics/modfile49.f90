! RUN: %python %S/test_modfile.py %s %flang_fc1
! Ensure that symbols and types needed to declare procedures and procedure pointers
! are properly imported into interfaces.
module m
  type :: t
  end type
  procedure(sin) :: ext
  interface
    subroutine subr(p1,p2)
      import ext, t
      procedure(ext) :: p1
      procedure(type(t)), pointer :: p2
    end subroutine
    function fun() result(res)
      import subr
      procedure(subr), pointer :: res
    end function
  end interface
end module

!Expect: m.mod
!module m
!type::t
!end type
!intrinsic::sin
!procedure(sin)::ext
!interface
!subroutine subr(p1,p2)
!import::ext
!import::t
!procedure(ext)::p1
!procedure(type(t)),pointer::p2
!end
!end interface
!interface
!function fun() result(res)
!import::subr
!procedure(subr),pointer::res
!end
!end interface
!end
