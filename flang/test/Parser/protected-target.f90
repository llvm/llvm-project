! RUN: %flang_fc1 -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -ffixed-form -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s --check-prefix=TREE

      module m
      integer, pointer, protected_target :: p
! CHECK: INTEGER, POINTER, PROTECTED_TARGET :: p
! TREE: AttrSpec -> ProtectedTarget
      type t
      integer, protected_target, pointer :: component
! CHECK: INTEGER, PROTECTED_TARGET, POINTER :: component
! TREE: ComponentAttrSpec -> ProtectedTarget
      end type
      contains
      subroutine s
      integer, pointer :: q, r
      protected_target q
! CHECK: PROTECTED_TARGET :: q
! TREE: ProtectedTargetStmt
      protected_target :: r
! CHECK: PROTECTED_TARGET :: r
! TREE: ProtectedTargetStmt
      end subroutine
      end module
