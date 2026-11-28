! RUN: %python %S/test_modfile.py %s %flang_fc1
! Preserve generic binding accessibility independently of the specific binding.

module m
  type t
  contains
    private
    procedure, nopass, public :: f
    generic, private :: g => f
    generic :: h => f
    generic, public :: j => f
  end type
contains
  subroutine f
  end subroutine
end module

!Expect: m.mod
!module m
!type::t
!contains
!procedure,nopass::f
!generic,private::g=>f
!generic,private::h=>f
!generic::j=>f
!end type
!contains
!subroutine f()
!end
!end
