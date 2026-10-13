! RUN: %python %S/test_modfile.py %s %flang_fc1
! PROTECTED_TARGET attribute (F2028 draft J3/26-007r1) in module files
module m
  real, pointer, protected_target :: p1
  real, pointer :: p2(:)
  protected_target :: p2
  type t
    real, pointer, protected_target :: c
  end type
 contains
  function f(a) result(r)
    real, pointer, protected_target, intent(in) :: a
    real, pointer, protected_target :: r
    r => a
  end
end

!Expect: m.mod
!module m
!real(4),pointer,protected_target::p1
!real(4),pointer,protected_target::p2(:)
!type::t
!real(4),pointer,protected_target::c
!end type
!contains
!function f(a) result(r)
!real(4),intent(in),pointer,protected_target::a
!real(4),pointer,protected_target::r
!end
!end
