! RUN: %flang_fc1 -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s | FileCheck --check-prefix=TREE %s

! The alignment directive is parsed among the components of a derived type.
! Semantics rejects it there for now (see Semantics/align-directive01.f90), so
! this test stops after parsing.

module m
  type t
    real :: x(8)
    !dir$ align(64) :: x
    ! CHECK: !DIR$ ALIGN(64) :: x
    ! TREE: ComponentDefStmt -> CompilerDirective -> Align
    ! TREE-NEXT: uint64_t = '64'
    ! TREE-NEXT: Name = 'x'
    real, allocatable :: y(:)
    !dir$ attributes align : 32 :: y
    ! CHECK: !DIR$ ALIGN(32) :: y
    ! TREE: ComponentDefStmt -> CompilerDirective -> Align
    ! TREE-NEXT: uint64_t = '32'
    ! TREE-NEXT: Name = 'y'
  end type
  ! CHECK: END TYPE
end module
