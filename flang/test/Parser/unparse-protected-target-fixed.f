! RUN: %flang_fc1 -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s

      subroutine test_fixed
        integer, pointer, protected_target :: p1
        integer, pointer :: p2
        integer, pointer :: p3
        protected_target :: p2
        protected_target p3
      end subroutine

! CHECK: SUBROUTINE test_fixed
! CHECK:  INTEGER, POINTER, PROTECTED_TARGET :: p1
! CHECK:  INTEGER, POINTER :: p2
! CHECK:  INTEGER, POINTER :: p3
! CHECK:  PROTECTED_TARGET :: p2
! CHECK:  PROTECTED_TARGET :: p3
! CHECK: END SUBROUTINE
