! RUN: %flang_fc1 -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s --check-prefix=TREE

module protected_target_syntax
  type box
    integer, pointer, protected_target :: p
    real, protected_target, pointer :: q(:)
  end type
  integer, pointer, protected_target :: p
  integer, pointer :: q, r
  protected_target :: q
  protected_target r
contains
  function get() result(p)
    integer, protected_target, pointer :: p
  end function
  subroutine keyword_name
    integer :: protected_target
    protected_target = 1
  end subroutine
end module

! CHECK: MODULE protected_target_syntax
! CHECK: TYPE :: box
! CHECK: INTEGER, POINTER, PROTECTED_TARGET :: p
! CHECK: REAL, PROTECTED_TARGET, POINTER :: q(:)
! CHECK: END TYPE
! CHECK: INTEGER, POINTER, PROTECTED_TARGET :: p
! CHECK: INTEGER, POINTER :: q, r
! CHECK: PROTECTED_TARGET :: q
! CHECK: PROTECTED_TARGET :: r
! CHECK: FUNCTION get() RESULT(p)
! CHECK: INTEGER, PROTECTED_TARGET, POINTER :: p
! CHECK: INTEGER protected_target
! CHECK: protected_target = 1

! TREE: ComponentAttrSpec -> ProtectedTarget
! TREE: ComponentAttrSpec -> ProtectedTarget
! TREE: AttrSpec -> ProtectedTarget
! TREE: ProtectedTargetStmt
! TREE: ProtectedTargetStmt
! TREE: AttrSpec -> ProtectedTarget
