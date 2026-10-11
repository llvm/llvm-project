! RUN: %python %S/test_modfile.py %s %flang_fc1
! RUN: mkdir -p %t
! RUN: %flang_fc1 -fsyntax-only -module-dir %t %s
! RUN: %python %S/test_errors.py %S/Inputs/modfile89-a.f90 %flang_fc1 -I %t

module protected_target_module
  integer, target :: storage
  integer, pointer, protected_target :: p
  type t
    integer, pointer, protected_target :: component
  end type
contains
  subroutine take(a)
    integer, pointer, protected_target, intent(inout) :: a
  end subroutine
  function get() result(r)
    integer, pointer, protected_target :: r
    r => storage
  end function
end module

!Expect: protected_target_module.mod
!module protected_target_module
!integer(4),target::storage
!integer(4),pointer,protected_target::p
!type::t
!integer(4),pointer,protected_target::component
!end type
!contains
!subroutine take(a)
!integer(4),intent(inout),pointer,protected_target::a
!end
!function get() result(r)
!integer(4),pointer,protected_target::r
!end
!end
