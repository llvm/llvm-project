! RUN: %flang_fc1 -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s --check-prefix=TREE
! Test parsing and unparsing of the PROTECTED_TARGET attribute, component
! attribute, and statement (F2028 draft J3/26-007r1).

module m
  type t
    real, pointer, protected_target :: c1
    real, protected_target, pointer :: c2(:)
  end type
  real, pointer, protected_target :: p1
  real, protected_target :: p2
  real, pointer :: p3, p4
  pointer :: p2
  protected_target :: p3
  protected_target p4
  real, pointer :: protected_target
  protected protected_target
 contains
  function f() result(r)
    real, pointer, protected_target :: r
    r => null()
  end function
end module

!CHECK: MODULE m
!CHECK:  TYPE :: t
!CHECK:   REAL, POINTER, PROTECTED_TARGET :: c1
!CHECK:   REAL, PROTECTED_TARGET, POINTER :: c2(:)
!CHECK:  END TYPE
!CHECK:  REAL, POINTER, PROTECTED_TARGET :: p1
!CHECK:  REAL, PROTECTED_TARGET :: p2
!CHECK:  REAL, POINTER :: p3, p4
!CHECK:  POINTER :: p2
!CHECK:  PROTECTED_TARGET :: p3
!CHECK:  PROTECTED_TARGET :: p4
!CHECK:  REAL, POINTER :: protected_target
!CHECK:  PROTECTED :: protected_target
!CHECK: CONTAINS
!CHECK:  FUNCTION f() RESULT(r)
!CHECK:   REAL, POINTER, PROTECTED_TARGET :: r

!TREE: ComponentAttrSpec -> ProtectedTarget
!TREE: AttrSpec -> ProtectedTarget
!TREE: OtherSpecificationStmt -> ProtectedTargetStmt -> Name = 'p3'
!TREE: OtherSpecificationStmt -> ProtectedTargetStmt -> Name = 'p4'
!TREE: OtherSpecificationStmt -> ProtectedStmt -> Name = 'protected_target'
