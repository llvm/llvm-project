! RUN: rm -rf %t && split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -J%t %t/m.f90
! RUN: %python %S/test_errors.py %t/use.f90 %flang_fc1 -J%t
! PROTECTED_TARGET attribute (F2028 draft J3/26-007r1): the constraints apply
! to entities and procedure characteristics read back from a module file.

!--- m.f90
module m
  real, target :: x = 1.
  real, pointer, protected_target :: p
  type t
    real, pointer :: q
    real, pointer, protected_target :: pq
  end type
 contains
  subroutine ptDummy(a)
    real, pointer, protected_target, intent(in) :: a
  end
  function ptResult()
    real, pointer, protected_target :: ptResult
    ptResult => x
  end
end

!--- use.f90
subroutine user
  use m
  interface
    subroutine plainDummy(a)
      real, pointer, intent(in) :: a
    end
  end interface
  real, pointer :: w
  real, pointer, protected_target :: q
  type(t) :: tt
  procedure(ptDummy), pointer :: pp
  p => x
  q => p
  q => ptResult()
  tt = t(w, p)
  call ptDummy(p)
  pp => ptDummy
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
  p = 2.
  !ERROR: Pointer 'w' must have the PROTECTED_TARGET attribute because its target is designated by PROTECTED_TARGET pointer 'p'
  w => p
  !ERROR: Pointer 'w' must have the PROTECTED_TARGET attribute because its target is designated by PROTECTED_TARGET pointer 'ptresult'
  w => ptResult()
  !ERROR: Pointer 'q' must have the PROTECTED_TARGET attribute because its target is designated by PROTECTED_TARGET pointer 'p'
  tt = t(p, p)
  !ERROR: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute because its actual argument is designated by PROTECTED_TARGET pointer 'p'
  call plainDummy(p)
  ! PROTECTED_TARGET is not a characteristic (F2028 draft 15.3.2.2).
  pp => plainDummy
end
