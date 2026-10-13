! RUN: %python %S/test_modfile.py %s %flang_fc1
! PROTECTED_TARGET is serialized for variables, components, dummies and results.

module m
  type box
    integer, pointer, protected_target :: component
  end type
  integer, pointer, protected_target :: p => null()
  integer, pointer :: q
  protected_target :: q
contains
  subroutine read_pointer(ptr)
    integer, pointer, protected_target, intent(in) :: ptr
  end subroutine
  function get() result(result)
    integer, pointer, protected_target :: result
    result => p
  end function
end module

!Expect: m.mod
!module m
!type::box
!integer(4),pointer,protected_target::component
!end type
!integer(4),pointer,protected_target::p
!intrinsic::null
!integer(4),pointer,protected_target::q
!contains
!subroutine read_pointer(ptr)
!integer(4),intent(in),pointer,protected_target::ptr
!end
!function get() result(result)
!integer(4),pointer,protected_target::result
!end
!end
