! RUN: %flang_fc1 -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s -check-prefix=TREE

! Test parsing of the F2028 PROTECTED_TARGET attribute and statement
! (J3/26-007r1 R749, R802, R863)

module m
  type t
    ! CHECK: REAL, POINTER, PROTECTED_TARGET :: c
    ! TREE: ComponentAttrSpec -> ProtectedTarget
    real, pointer, protected_target :: c
  end type
  ! CHECK: REAL, PROTECTED_TARGET, POINTER :: p1
  ! TREE: AttrSpec -> ProtectedTarget
  real, protected_target, pointer :: p1
  ! PROTECTED is a prefix of PROTECTED_TARGET.
  ! CHECK: REAL, POINTER, PROTECTED, PROTECTED_TARGET :: p2
  ! TREE: AttrSpec -> Protected
  ! TREE: AttrSpec -> ProtectedTarget
  real, pointer, protected, protected_target :: p2
  real, pointer :: p3, p4, p5, p6
  ! CHECK: PROTECTED_TARGET :: p3
  ! TREE: ProtectedTargetStmt
  ! TREE: Name = 'p3'
  protected_target :: p3
  ! CHECK: PROTECTED_TARGET :: p4, p5
  ! TREE: ProtectedTargetStmt
  ! TREE: Name = 'p4'
  ! TREE: Name = 'p5'
  protected_target p4, p5
  ! CHECK: PROTECTED :: p6
  ! TREE: ProtectedStmt
  ! TREE: Name = 'p6'
  protected p6
end module
