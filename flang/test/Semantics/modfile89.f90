! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -J%t %t/m.f90
! RUN: FileCheck %s --check-prefix=MOD < %t/protected_target_mod.mod
! RUN: %flang_fc1 -fsyntax-only -J%t %t/valid.f90
! RUN: %python %S/test_errors.py %t/invalid.f90 %flang_fc1 -J%t
! Verify writing and reading attributes on objects, components, dummies and results.

!--- m.f90
module protected_target_mod
  integer, target :: x
  integer, pointer, protected_target :: p
  type t
    integer, pointer, protected_target :: p
  end type
contains
  subroutine read_pointer(q)
    integer, pointer, protected_target, intent(in) :: q
  end subroutine
  function get() result(r)
    integer, pointer, protected_target :: r
    r => x
  end function
end module
! MOD: integer(4),pointer,protected_target::p
! MOD: type::t
! MOD: integer(4),pointer,protected_target::p
! MOD: subroutine read_pointer(q)
! MOD: integer(4),intent(in),pointer,protected_target::q
! MOD: function get() result(r)
! MOD: integer(4),pointer,protected_target::r

!--- valid.f90
subroutine valid
  use protected_target_mod
  type(t) :: obj
  integer, pointer, protected_target :: q
  p => x
  q => p
  q => get()
  obj = t(p)
  call read_pointer(p)
  print *, p, obj%p
  nullify(p)
end subroutine

!--- invalid.f90
subroutine invalid
  use protected_target_mod, only: t, get, renamed => p
  type(t) :: obj
  integer, pointer :: q
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: Target of PROTECTED_TARGET pointer 'renamed' is not definable
  renamed = 1
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
  obj%p = 1
  !ERROR: PROTECTED_TARGET pointer 'renamed' or a subobject may not be associated with pointer 'q' without PROTECTED_TARGET
  q => renamed
  !ERROR: The PROTECTED_TARGET result of 'get' or a subobject may not be associated with pointer 'q' without PROTECTED_TARGET
  q => get()
end subroutine
