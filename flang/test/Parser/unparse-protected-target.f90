! RUN: %flang_fc1 -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s

subroutine test_unparse()
  type t
    integer, pointer, protected_target :: c
  end type
  integer, pointer, protected_target :: p1
  integer, protected :: prot_var
  integer, pointer :: p2
  integer, pointer :: p3
  protected_target :: p2
  protected_target p3
  protected :: prot_var2
end subroutine

! CHECK: SUBROUTINE test_unparse
! CHECK:  TYPE :: t
! CHECK:   INTEGER, POINTER, PROTECTED_TARGET :: c
! CHECK:  END TYPE
! CHECK:  INTEGER, POINTER, PROTECTED_TARGET :: p1
! CHECK:  INTEGER, PROTECTED :: prot_var
! CHECK:  INTEGER, POINTER :: p2
! CHECK:  INTEGER, POINTER :: p3
! CHECK:  PROTECTED_TARGET :: p2
! CHECK:  PROTECTED_TARGET :: p3
! CHECK:  PROTECTED :: prot_var2
! CHECK: END SUBROUTINE
