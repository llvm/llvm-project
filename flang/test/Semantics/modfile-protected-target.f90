! RUN: %python %S/test_modfile.py %s %flang_fc1
! Test module file emission of the F2028 PROTECTED_TARGET attribute
module m
  type t
    real, pointer, protected_target :: c
  end type
  real, pointer, protected_target :: p
  real, pointer :: q(:)
  protected_target :: q
 contains
  function f(a) result(r)
    real, pointer, protected_target, intent(in) :: a
    real, pointer, protected_target :: r
    r => a
  end
end

!Expect: m.mod
!module m
!type::t
!real(4),pointer,protected_target::c
!end type
!real(4),pointer,protected_target::p
!real(4),pointer,protected_target::q(:)
!contains
!function f(a) result(r)
!real(4),intent(in),pointer,protected_target::a
!real(4),pointer,protected_target::r
!end
!end
