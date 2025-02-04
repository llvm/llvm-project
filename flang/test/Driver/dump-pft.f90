! RUN: %flang_fc1 -fdebug-dump-parse-tree %s 2>&1 | FileCheck %s --check-prefix=PARSE_TREE
! RUN: %flang_fc1 -fdebug-dump-pft %s 2>&1 | FileCheck %s --check-prefix=PFT
! RUN: bbc -pft-test %s 2>&1 | FileCheck %s --check-prefix=PFT

! PFT: 1 Subroutine test_routine: subroutine test_routine(a, b, n)
! PFT-NEXT:  1 EndSubroutineStmt: end subroutine
! PRF-NEXT: End Subroutine test_routine
! PFT-NOT: Program -> ProgramUnit -> SubroutineSubprogram

! PARSE_TREE: Program -> ProgramUnit -> SubroutineSubprogram
! PARSE_TREE-NEXT: | SubroutineStmt
! PARSE_TREE-NEXT: | | Name = 'test_routine'
! PARSE_TREE-NEXT: | | DummyArg -> Name = 'a'
! PARSE_TREE-NEXT: | | DummyArg -> Name = 'b'
! PARSE_TREE-NEXT: | | DummyArg -> Name = 'n'
! PARSE_TREE-NEXT: | SpecificationPart
! PARSE_TREE-NEXT: | | ImplicitPart ->
! PARSE_TREE-NEXT: | ExecutionPart -> Block
! PARSE_TREE-NEXT: | EndSubroutineStmt ->
! PARSE_TREE-NOT: Subroutine test_routine: subroutine test_routine(a, b, n)

subroutine test_routine(a, b, n)
end subroutine
