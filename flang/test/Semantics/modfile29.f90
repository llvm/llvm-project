! RUN: %python %S/test_modfile.py %s %flang_fc1
! Check that implicitly typed entities get a type in the module file.

module m
  public :: a
  private :: b
  protected :: i
  allocatable :: j
end

!Expect: m.mod
!module m
! real(4)::a
! real(4),private::b
! integer(4),protected::i
! integer(4),allocatable::j
!end

module mpt
  integer, target :: x
  integer, pointer, protected_target :: p
contains
  subroutine take(a)
    integer, pointer, protected_target, intent(in) :: a
  end subroutine
  function f() result(r)
    integer, pointer, protected_target :: r
  end function
end module

!Expect: mpt.mod
!module mpt
!integer(4),target::x
!integer(4),pointer,protected_target::p
!contains
!subroutine take(a)
!integer(4),intent(in),pointer,protected_target::a
!end
!function f() result(r)
!integer(4),pointer,protected_target::r
!end
!end
