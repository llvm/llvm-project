!RUN: rm -rf %t && mkdir -p %t
!RUN: %flang_fc1 -fsyntax-only -DSTEP=1 -J%t %s
!RUN: not %flang_fc1 -fsyntax-only -DSTEP=2 -J%t %s 2>&1 | FileCheck %s

! The F2028 PROTECTED_TARGET attribute must survive being written to a module
! file and read back, so that its constraints apply to USE-associated
! pointers, components, and procedure interfaces.  The module is compiled in a
! separate invocation, so the second one reads it from its module file.

#if STEP == 1
module pt_readback
  type t
    real, pointer, protected_target :: c
  end type
  real, target :: tgt
  real, pointer, protected_target :: p
 contains
  subroutine take(a)
    real, pointer, protected_target, intent(in) :: a
  end
  function get()
    real, pointer, protected_target :: get
    get => tgt
  end
end
#else
subroutine s
  use pt_readback
  real, pointer :: w
  type(t) :: x
  interface
    subroutine ordinary(a)
      real, pointer, intent(in) :: a
    end
  end interface
  procedure(take), pointer :: pp
  p => tgt
  x%c => tgt
  call take(p)
  pp => take
  pp => ordinary
  call pp(p)
!CHECK-NOT: incompatible
!CHECK-NOT: error: POINTER dummy argument
!CHECK: error: Left-hand side of assignment is not definable
!CHECK: because: 'p' is a PROTECTED_TARGET pointer whose target may not be defined through it
  p = 1.
!CHECK: error: Left-hand side of assignment is not definable
!CHECK: because: 'c' is a PROTECTED_TARGET pointer whose target may not be defined through it
  x%c = 1.
!CHECK: error: Pointer 'w' must have the PROTECTED_TARGET attribute to be associated with the target of PROTECTED_TARGET pointer 'get'
  w => get()
!CHECK: error: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute to be associated with the target of PROTECTED_TARGET pointer 'p'
  call ordinary(p)
end
#endif
