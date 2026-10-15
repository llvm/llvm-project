! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -J%t %t/m.f90
! RUN: FileCheck %s --check-prefix=MOD < %t/protected_target_mod.mod
! RUN: %flang_fc1 -fsyntax-only -I%t -J%t %t/good.f90
! RUN: %python %S/test_errors.py %t/bad.f90 %flang_fc1 -I%t

!--- m.f90
module protected_target_mod
  integer, pointer, protected_target :: p
  type t
    integer, pointer, protected_target :: component
  end type
contains
  subroutine read_pointer(x)
    integer, pointer, protected_target, intent(in) :: x
  end subroutine
  function get() result(r)
    integer, pointer, protected_target :: r
    r => p
  end function
end module

! MOD-DAG: integer(4),pointer,protected_target::p
! MOD-DAG: integer(4),pointer,protected_target::component
! MOD-DAG: integer(4),intent(in),pointer,protected_target::x
! MOD-DAG: integer(4),pointer,protected_target::r

!--- good.f90
subroutine good
  use protected_target_mod
  integer, pointer, protected_target :: q
  type(t) :: object
  q => p
  q => get()
  object = t(p)
  call read_pointer(p)
  nullify(p, q)
end subroutine

!--- bad.f90
subroutine bad
  use protected_target_mod, only: renamed => p, get, t
  integer, pointer :: ordinary
  type(t) :: object
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'p' is not definable
  renamed = 1
  !ERROR: pointer 'ordinary' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'p'
  ordinary => renamed
  !ERROR: pointer 'ordinary' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'r'
  ordinary => get()
  !ERROR: Allocation of a PROTECTED_TARGET pointer requires SOURCE=
  allocate(renamed)
  !ERROR: A PROTECTED_TARGET pointer may not be deallocated
  deallocate(renamed)
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'component' is not definable
  object%component = 1
end subroutine
