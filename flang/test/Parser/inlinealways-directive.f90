! Test that the INLINEALWAYS directive can be parsed, both at callsite and within the function

! RUN: %flang_fc1 -fdebug-dump-parse-tree %s 2>&1 | FileCheck %s --check-prefix=PARSE-TREE
! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s --check-prefix=UNPARSE

subroutine test_function()
  !DIR$ INLINEALWAYS test_function
end subroutine

subroutine test_function2()
end subroutine

subroutine test()
  call test_function()
  !DIR$ INLINEALWAYS
  call test_function2()
end subroutine

! UNPARSE: SUBROUTINE test_function
! UNPARSE:  !DIR$ INLINEALWAYS TEST_FUNCTION
! UNPARSE: END SUBROUTINE
! UNPARSE: SUBROUTINE test_function2
! UNPARSE: END SUBROUTINE
! UNPARSE: SUBROUTINE test
! UNPARSE:   CALL test_function()
! UNPARSE:  !DIR$ INLINEALWAYS
! UNPARSE:   CALL test_function2()
! UNPARSE: END SUBROUTINE

! PARSE-TREE: Program -> ProgramUnit -> SubroutineSubprogram
! PARSE-TREE: | SubroutineStmt
! PARSE-TREE: | | Name = 'test_function'
! PARSE-TREE: | SpecificationPart
! PARSE-TREE: | | ImplicitPart -> 
! PARSE-TREE: | ExecutionPart -> Block
! PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> CompilerDirective -> InlineAlways -> Name = 'test_function'
! PARSE-TREE: | EndSubroutineStmt -> 
! PARSE-TREE: ProgramUnit -> SubroutineSubprogram
! PARSE-TREE: | SubroutineStmt
! PARSE-TREE: | | Name = 'test_function2'
! PARSE-TREE: | SpecificationPart
! PARSE-TREE: | | ImplicitPart -> 
! PARSE-TREE: | ExecutionPart -> Block
! PARSE-TREE: | EndSubroutineStmt -> 
! PARSE-TREE: ProgramUnit -> SubroutineSubprogram
! PARSE-TREE: | SubroutineStmt
! PARSE-TREE: | | Name = 'test'
! PARSE-TREE: | SpecificationPart
! PARSE-TREE: | | ImplicitPart -> 
! PARSE-TREE: | ExecutionPart -> Block
! PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> CallStmt = 'CALL test_function()'
! PARSE-TREE: | | | Call
! PARSE-TREE: | | | | ProcedureDesignator -> Name = 'test_function'
! PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> CompilerDirective -> InlineAlways -> 
! PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> CallStmt = 'CALL test_function2()'
! PARSE-TREE: | | | Call
! PARSE-TREE: | | | | ProcedureDesignator -> Name = 'test_function2'
! PARSE-TREE: | EndSubroutineStmt -> 
