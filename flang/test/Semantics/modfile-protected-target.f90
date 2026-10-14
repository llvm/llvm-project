! Test modfile emission and readback for PROTECTED_TARGET attribute
! RUN: %python %S/test_modfile.py %s %flang_fc1

module m1
  type :: t
    integer, pointer, protected_target :: c
  end type
  integer, pointer, protected_target :: p
contains
  subroutine sub(arg)
    integer, pointer, protected_target :: arg
  end subroutine
  function func() result(r)
    integer, pointer, protected_target :: r
  end function
end module m1

!Expect: m1.mod
!module m1
!type::t
!integer(4),pointer,protected_target::c
!end type
!integer(4),pointer,protected_target::p
!contains
!subroutine sub(arg)
!integer(4),pointer,protected_target::arg
!end
!function func() result(r)
!integer(4),pointer,protected_target::r
!end
!end
