! RUN: %flang_fc1 -fdebug-dump-parse-tree %s | FileCheck %s --check-prefix=TREE
! RUN: %flang_fc1 -fdebug-unparse-no-sema %s | FileCheck %s --check-prefix=UNPARSE
! RUN: %flang_fc1 -ffixed-form -fdebug-unparse-no-sema %s | FileCheck %s --check-prefix=UNPARSE

      module m
      integer, pointer, protected_target :: p
      integer, protected, target :: x
      type t
        integer, PrOtEcTeD_TaRgEt, pointer :: q
      end type
      contains
      subroutine s
      integer, pointer :: a, b
      protected_target a
      protected_target :: b
      integer :: protected_target
      protected_target = 1
      end subroutine
      end module

! TREE: ProtectedTarget
! TREE: Protected
! TREE: ComponentAttrSpec
! TREE: ProtectedTarget
! TREE: ProtectedTargetStmt
! TREE: ProtectedTargetStmt

! UNPARSE: INTEGER, POINTER, PROTECTED_TARGET :: p
! UNPARSE: INTEGER, PROTECTED, TARGET :: x
! UNPARSE: INTEGER, PROTECTED_TARGET, POINTER :: q
! UNPARSE: PROTECTED_TARGET :: a
! UNPARSE: PROTECTED_TARGET :: b
! UNPARSE: INTEGER protected_target
! UNPARSE: protected_target = 1
