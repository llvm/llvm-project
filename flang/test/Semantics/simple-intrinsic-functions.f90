! Test that intrinsic functions are classified SIMPLE (F2023 16.1)
! RUN: %flang_fc1 -fsyntax-only %s

! An intrinsic function is SIMPLE and can target a SIMPLE procedure pointer
module simple_intrinsic_function
  implicit none
  abstract interface
    simple real function ifc(x)
      real, intent(in) :: x
    end function
  end interface
contains
  subroutine test()
    procedure(ifc), pointer :: sp
    intrinsic :: sin
    sp => sin
  end subroutine
end module
