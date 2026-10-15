! RUN: %flang_fc1 -ffixed-form -fdebug-unparse-no-sema %s | FileCheck %s
      subroutine s
      integer p,q
      pointer p,q
      protected_target p
      protected_target::q
      end
! CHECK: PROTECTED_TARGET :: p
! CHECK: PROTECTED_TARGET :: q
