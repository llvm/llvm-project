! RUN: %flang_fc1 -fdebug-dump-parse-tree %s | FileCheck %s --check-prefix=TREE
! RUN: %flang_fc1 -fdebug-unparse-no-sema %s | FileCheck %s --check-prefix=UNPARSE

module protected_target_syntax
  integer, pointer, protected_target :: p
  type :: t
    integer, protected_target, pointer :: component
  end type
  integer, pointer :: q, r
  protected_target :: q
  protected_target r
end module

subroutine keyword_as_name()
  integer :: protected_target
  protected_target = 1
end subroutine

! TREE: AttrSpec -> ProtectedTarget
! TREE: ComponentAttrSpec -> ProtectedTarget
! TREE: OtherSpecificationStmt -> ProtectedTargetStmt
! TREE: OtherSpecificationStmt -> ProtectedTargetStmt

! UNPARSE: INTEGER, POINTER, PROTECTED_TARGET :: p
! UNPARSE: INTEGER, PROTECTED_TARGET, POINTER :: component
! UNPARSE: PROTECTED_TARGET :: q
! UNPARSE: PROTECTED_TARGET :: r
! UNPARSE: INTEGER protected_target
! UNPARSE: protected_target = 1
