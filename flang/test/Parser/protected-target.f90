! RUN: %flang_fc1 -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s -check-prefix=TREE

! Test parsing of the PROTECTED_TARGET attribute and statement
! (F2028 R802, R749, R863)

module m
  !CHECK: TYPE :: t
  !CHECK: INTEGER, POINTER, PROTECTED_TARGET :: c
  !TREE: ComponentAttrSpec -> Pointer
  !TREE: ComponentAttrSpec -> ProtectedTarget
  type :: t
    integer, pointer, protected_target :: c
  end type
  !CHECK: REAL, POINTER, PROTECTED_TARGET :: p1
  !TREE: AttrSpec -> Pointer
  !TREE: AttrSpec -> ProtectedTarget
  real, pointer, protected_target :: p1
  !CHECK: REAL, PROTECTED_TARGET, POINTER :: p2
  real, protected_target, pointer :: p2
  !CHECK: REAL, POINTER :: p3, p4, p5
  real, pointer :: p3, p4, p5
  !CHECK: PROTECTED_TARGET :: p3
  !TREE: OtherSpecificationStmt -> ProtectedTargetStmt -> Name = 'p3'
  protected_target :: p3
  !CHECK: PROTECTED_TARGET :: p4, p5
  !TREE: OtherSpecificationStmt -> ProtectedTargetStmt -> Name = 'p4'
  !TREE-NEXT: Name = 'p5'
  protected_target p4, p5
  !CHECK: REAL, PROTECTED, TARGET :: x
  !TREE: AttrSpec -> Protected
  real, protected, target :: x
  !CHECK: REAL, TARGET :: y
  !CHECK: PROTECTED :: y
  !TREE: OtherSpecificationStmt -> ProtectedStmt -> Name = 'y'
  real, target :: y
  protected y
end
