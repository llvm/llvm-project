! RUN: %flang_fc1 -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s --check-prefix=UNPARSE
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s --check-prefix=TREE

module m
  type t
    ! UNPARSE: INTEGER, POINTER, PROTECTED_TARGET :: c
    ! TREE: ComponentAttrSpec -> ProtectedTarget
    integer, pointer, protected_target :: c
  end type
  ! UNPARSE: INTEGER, PROTECTED_TARGET, POINTER :: p
  ! TREE: AttrSpec -> ProtectedTarget
  integer, PROTECTED_TARGET, pointer :: p
  integer, pointer :: q, r, s
  ! UNPARSE: PROTECTED_TARGET :: q, r
  ! TREE: ProtectedTargetStmt
  protected_target q, r
  ! UNPARSE: PROTECTED_TARGET :: s
  ! TREE: ProtectedTargetStmt
  protected_target :: s
contains
  function f() result(x)
    ! UNPARSE: INTEGER, POINTER, PROTECTED_TARGET :: x
    ! TREE: AttrSpec -> ProtectedTarget
    integer, pointer, protected_target :: x
  end function
end module
