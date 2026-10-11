! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -J%t %t/mod.f90
! RUN: %python %S/test_errors.py %t/use.f90 %flang_fc1 -I%t

!--- mod.f90
module modfile_protected_target_use
  implicit none
  integer, target, save :: x = 1
  integer, pointer, protected_target, save :: p => x
  type holder
    integer, pointer, protected_target :: c => x
  end type
contains
  subroutine takes_plain(a)
    integer, pointer :: a
  end
  subroutine takes_pt(a)
    integer, pointer, protected_target :: a
  end
  subroutine takes_no_intent(a)
    integer :: a
  end
  function plain_result() result(r)
    integer, pointer :: r
    r => x
  end
  function pt_result() result(r)
    integer, pointer, protected_target :: r
    r => x
  end
end module

!--- use.f90
subroutine use_it
  use modfile_protected_target_use
  implicit none
  procedure(plain_result), pointer :: pp
  procedure(pt_result), pointer :: qq
  type(holder) :: obj
  integer, pointer :: q
  call takes_pt(p)
  !ERROR: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute when the actual argument is a PROTECTED_TARGET pointer or a subobject of one
  call takes_plain(p)
  qq => pt_result
  pp => pt_result
  !ERROR: Pointer assignment target is a PROTECTED_TARGET pointer or a subobject of one; the left-hand side pointer must also have the PROTECTED_TARGET attribute
  q => obj%c
  !ERROR: Non-POINTER dummy argument 'a=' must have INTENT(IN) when the actual argument is a PROTECTED_TARGET pointer or a subobject of one
  call takes_no_intent(obj%c)
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'p' is a PROTECTED_TARGET pointer, so its target may not be defined through it
  p = 2
end

subroutine use_rename
  use modfile_protected_target_use, only: rp => p
  implicit none
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'rp' is a PROTECTED_TARGET pointer, so its target may not be defined through it
  rp = 3
end
