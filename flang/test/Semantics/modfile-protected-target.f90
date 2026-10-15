! RUN: %python %S/test_modfile.py %s %flang_fc1
! RUN: rm -rf %t && mkdir -p %t
! RUN: %flang_fc1 -fsyntax-only -module-dir %t %s
! RUN: %python %S/test_errors.py %S/Inputs/protected-target-modfile.f90 %flang_fc1 -I %t
! Check emission and reading of object, component, dummy and result attributes.

module protected_target_mod
  type holder
    integer, pointer, protected_target :: q
  end type
  integer, pointer :: p
  protected_target p
  integer, protected, target :: x
contains
  subroutine read_pointer(a)
    integer, pointer, protected_target, intent(in) :: a
  end subroutine
  subroutine ordinary_pointer(a)
    integer, pointer, intent(in) :: a
  end subroutine
  subroutine default_value(a)
    integer :: a
  end subroutine
  function result_pointer() result(res)
    integer, pointer, protected_target :: res
    integer, target, save :: target = 1
    res => target
  end function
end module

!Expect: protected_target_mod.mod
!module protected_target_mod
!type::holder
!integer(4),pointer,protected_target::q
!end type
!integer(4),pointer,protected_target::p
!integer(4),protected,target::x
!contains
!subroutine read_pointer(a)
!integer(4),intent(in),pointer,protected_target::a
!end
!subroutine ordinary_pointer(a)
!integer(4),intent(in),pointer::a
!end
!subroutine default_value(a)
!integer(4)::a
!end
!function result_pointer() result(res)
!integer(4),pointer,protected_target::res
!end
!end
