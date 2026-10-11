! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -module-dir %t %t/m.f90
! RUN: FileCheck %s --check-prefix=MOD < %t/pt_module.mod
! RUN: %flang_fc1 -fsyntax-only -Wpointer-to-undefinable -module-dir %t %t/valid.f90
! RUN: not %flang_fc1 -fsyntax-only -module-dir %t %t/invalid.f90 2>&1 | FileCheck %s --implicit-check-not=error:

!--- m.f90
module pt_module
  integer, target, protected :: x
  integer, pointer, protected_target :: p
  type t
    integer, pointer, protected_target :: c
  end type
contains
  subroutine read_pointer(a)
    integer, pointer, protected_target, intent(in) :: a
  end
  function get() result(r)
    integer, pointer, protected_target :: r
    r => x
  end
end

!--- valid.f90
subroutine valid
  use pt_module
  type(t) :: obj
  integer, pointer :: w
  p => x
  p => get()
  obj = t(p)
  call read_pointer(p)
  call read_pointer(w)
  print *, p, obj%c
  nullify(p)
end

!--- invalid.f90
subroutine invalid
  use pt_module
  type(t) :: obj
  integer, pointer :: w
  p = 1
  obj%c = 1
  w => get()
end

! MOD-DAG: integer(4),pointer,protected_target::p
! MOD-DAG: integer(4),pointer,protected_target::c
! MOD-DAG: integer(4),intent(in),pointer,protected_target::a
! MOD-DAG: integer(4),pointer,protected_target::r
! CHECK: error: Semantic errors in
! CHECK: error: Left-hand side of assignment is not definable
! CHECK: because: The target of PROTECTED_TARGET pointer 'p' is not definable
! CHECK: error: Left-hand side of assignment is not definable
! CHECK: because: The target of PROTECTED_TARGET pointer 'c' is not definable
! CHECK: error: PROTECTED_TARGET pointer 'r' or its subobject may not be associated with pointer 'w' without PROTECTED_TARGET
