! RUN: mkdir -p %t.mod
! RUN: %flang_fc1 -fsyntax-only -module-dir %t.mod %S/Inputs/modfile-protected-target-a.f90
! RUN: %python %S/test_errors.py %s %flang_fc1 -I%t.mod
! Check constraints after reading PROTECTED_TARGET attributes from a .mod file.

subroutine module_reader
  use pt_mod
  implicit none
  type plain
    integer, pointer :: q
  end type
  type(holder) :: object
  type(plain) :: ordinary_object
  integer, pointer :: w
  integer, pointer, protected_target :: keep
  external implicit_sub
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable via that pointer
  p = 1
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: Target of PROTECTED_TARGET pointer 'q' is not definable via that pointer
  object%q = 1
  !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be associated with pointer 'w' without PROTECTED_TARGET
  w => p
  !ERROR: pointer 'w' may not be associated with the PROTECTED_TARGET result of function 'get' without PROTECTED_TARGET
  w => get()
  !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be associated with pointer 'q' without PROTECTED_TARGET
  ordinary_object = plain(p)
  !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be associated with dummy argument 'a=' without PROTECTED_TARGET
  call ordinary_pointer(p)
  !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may be associated only with an INTENT(IN) nonpointer dummy argument 'a='
  call default_intent(p)
  !ERROR: PROTECTED_TARGET pointer 'p' or its subobject requires an explicit interface
  call implicit_sub(p)
  !ERROR: PROTECTED_TARGET pointer 'p' in ALLOCATE requires a SOURCE= clause
  allocate(p)
  !ERROR: PROTECTED_TARGET pointer 'q' may not appear in DEALLOCATE
  deallocate(object%q)
  keep => p
  keep => get()
  object = holder(p)
  call read_pt(p)
  call read_value(p)
end
