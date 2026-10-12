! RUN: %flang_fc1 -fdebug-unparse-no-sema %s | FileCheck %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s | FileCheck %s --check-prefix=TREE

module protected_target_syntax
  type t
    ! CHECK: INTEGER, PROTECTED_TARGET, POINTER :: p
    integer, protected_target, pointer :: p
  end type
  ! CHECK: INTEGER, POINTER, PROTECTED_TARGET :: p
  integer, pointer, protected_target :: p
  integer, pointer :: q, r, s
  ! CHECK: PROTECTED_TARGET :: q
  protected_target :: q
  ! CHECK: PROTECTED_TARGET :: r, s
  protected_target r, s
end

! TREE: ComponentAttrSpec -> ProtectedTarget
! TREE: AttrSpec -> ProtectedTarget
! TREE: ProtectedTargetStmt
! TREE: ProtectedTargetStmt
