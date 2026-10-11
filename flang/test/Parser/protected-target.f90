! Test PROTECTED_TARGET parsing and unparsing.
! RUN: %flang_fc1 -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s

subroutine stmt_forms(p, q)
  integer, pointer :: p
  integer, pointer, protected_target :: q
  protected_target p
  protected_target :: q
  protected :: p
end

module m
  type t
    integer, pointer, protected_target :: c
  end type
end module

! CHECK: SUBROUTINE stmt_forms
! CHECK: INTEGER, POINTER :: p
! CHECK: INTEGER, POINTER, PROTECTED_TARGET :: q
! CHECK: PROTECTED_TARGET :: p
! CHECK: PROTECTED_TARGET :: q
! CHECK: PROTECTED :: p
! CHECK: END SUBROUTINE
! CHECK: MODULE m
! CHECK: TYPE :: t
! CHECK: INTEGER, POINTER, PROTECTED_TARGET :: c
! CHECK: END TYPE
! CHECK: END MODULE
