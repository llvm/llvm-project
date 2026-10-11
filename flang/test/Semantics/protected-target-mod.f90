! RUN: %python %S/test_modfile.py %s %flang_fc1
! The PROTECTED_TARGET attribute in module files

module m
  integer, target :: x
  integer, pointer, protected_target :: p1 => x
  real, pointer :: p2(:)
  protected_target :: p2
  type :: t
    integer, pointer, protected_target :: c
  end type
 contains
  function f(d) result(r)
    integer, pointer, protected_target, intent(in) :: d
    integer, pointer, protected_target :: r
    r => d
  end
end

!Expect: m.mod
!module m
!integer(4),target::x
!integer(4),pointer,protected_target::p1
!real(4),pointer,protected_target::p2(:)
!type::t
!integer(4),pointer,protected_target::c
!end type
!contains
!function f(d) result(r)
!integer(4),intent(in),pointer,protected_target::d
!integer(4),pointer,protected_target::r
!end
!end
