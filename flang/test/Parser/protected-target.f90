! RUN: %flang_fc1 -fdebug-unparse-no-sema %s | FileCheck %s --check-prefix=UNPARSE
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s | FileCheck %s --check-prefix=TREE

module m
  integer, pointer, protected_target :: p
  type t
    integer, protected_target, pointer :: c
  end type
contains
  subroutine s
    integer, pointer :: a, b
    protected_target a
    protected_target :: b
  end subroutine
end module

! UNPARSE: INTEGER, POINTER, PROTECTED_TARGET :: p
! UNPARSE: INTEGER, PROTECTED_TARGET, POINTER :: c
! UNPARSE: PROTECTED_TARGET :: a
! UNPARSE: PROTECTED_TARGET :: b
! TREE: AttrSpec -> ProtectedTarget
! TREE: ComponentAttrSpec -> ProtectedTarget
! TREE: ProtectedTargetStmt
! TREE: ProtectedTargetStmt
