! Test that the INLINEALWAYS directive can be parsed, both at callsite and within the function

! RUN: %flang_fc1 -fdebug-dump-parse-tree %s 2>&1 | FileCheck %s --check-prefix=PARSE-TREE
! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s --check-prefix=UNPARSE

subroutine test_subroutine()
  !DIR$ INLINEALWAYS test_subroutine
end subroutine

subroutine test_subroutine2()
end subroutine

integer function test_func() result(res)
  !DIR$ INLINEALWAYS test_func
  res = 10
end function

integer function test_func2() result(res)
  res = 10
end function

subroutine test()
  implicit none
  integer :: i, test_func1, test_func2

  call test_subroutine()
  !DIR$ INLINEALWAYS
  call test_subroutine2()

  i = test_func1()
  !DIR$ INLINEALWAYS
  i = test_func2()
end subroutine

! UNPARSE: SUBROUTINE test_subroutine
! UNPARSE:  !DIR$ INLINEALWAYS TEST_SUBROUTINE
! UNPARSE: END SUBROUTINE
! UNPARSE: SUBROUTINE test_subroutine2
! UNPARSE: END SUBROUTINE
! UNPARSE: INTEGER FUNCTION test_func() RESULT(res)
! UNPARSE:  !DIR$ INLINEALWAYS TEST_FUNC
! UNPARSE:   res=10_4
! UNPARSE: END FUNCTION
! UNPARSE: INTEGER FUNCTION test_func2() RESULT(res)
! UNPARSE:   res=10_4
! UNPARSE: END FUNCTION
! UNPARSE: SUBROUTINE test
! UNPARSE:  IMPLICIT NONE
! UNPARSE:  INTEGER i, test_func1, test_func2
! UNPARSE:   CALL test_subroutine()
! UNPARSE:  !DIR$ INLINEALWAYS
! UNPARSE:   CALL test_subroutine2()
! UNPARSE:   i=test_func1()
! UNPARSE:  !DIR$ INLINEALWAYS
! UNPARSE:   i=test_func2()
! UNPARSE: END SUBROUTINE

! PARSE-TREE: ======================== Flang: parse tree dump ========================
! PARSE-TREE: Program -> ProgramUnit -> SubroutineSubprogram
! PARSE-TREE: | SubroutineStmt
! PARSE-TREE: | | Name = 'test_subroutine'
! PARSE-TREE: | SpecificationPart
! PARSE-TREE: | | ImplicitPart -> 
! PARSE-TREE: | ExecutionPart -> Block
! PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> CompilerDirective -> InlineAlways -> Name = 'test_subroutine'
! PARSE-TREE: | EndSubroutineStmt -> 
! PARSE-TREE: ProgramUnit -> SubroutineSubprogram
! PARSE-TREE: | SubroutineStmt
! PARSE-TREE: | | Name = 'test_subroutine2'
! PARSE-TREE: | SpecificationPart
! PARSE-TREE: | | ImplicitPart -> 
! PARSE-TREE: | ExecutionPart -> Block
! PARSE-TREE: | EndSubroutineStmt -> 
! PARSE-TREE: ProgramUnit -> FunctionSubprogram
! PARSE-TREE: | FunctionStmt
! PARSE-TREE: | | PrefixSpec -> DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec -> 
! PARSE-TREE: | | Name = 'test_func'
! PARSE-TREE: | | Suffix
! PARSE-TREE: | | | Name = 'res'
! PARSE-TREE: | SpecificationPart
! PARSE-TREE: | | ImplicitPart -> 
! PARSE-TREE: | ExecutionPart -> Block
! PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> CompilerDirective -> InlineAlways -> Name = 'test_func'
! PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'res=10_4'
! PARSE-TREE: | | | Variable = 'res'
! PARSE-TREE: | | | | Designator -> DataRef -> Name = 'res'
! PARSE-TREE: | | | Expr = '10_4'
! PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '10'
! PARSE-TREE: | EndFunctionStmt -> 
! PARSE-TREE: ProgramUnit -> FunctionSubprogram
! PARSE-TREE: | FunctionStmt
! PARSE-TREE: | | PrefixSpec -> DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec -> 
! PARSE-TREE: | | Name = 'test_func2'
! PARSE-TREE: | | Suffix
! PARSE-TREE: | | | Name = 'res'
! PARSE-TREE: | SpecificationPart
! PARSE-TREE: | | ImplicitPart -> 
! PARSE-TREE: | ExecutionPart -> Block
! PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'res=10_4'
! PARSE-TREE: | | | Variable = 'res'
! PARSE-TREE: | | | | Designator -> DataRef -> Name = 'res'
! PARSE-TREE: | | | Expr = '10_4'
! PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '10'
! PARSE-TREE: | EndFunctionStmt -> 
! PARSE-TREE: ProgramUnit -> SubroutineSubprogram
! PARSE-TREE: | SubroutineStmt
! PARSE-TREE: | | Name = 'test'
! PARSE-TREE: | SpecificationPart
! PARSE-TREE: | | ImplicitPart -> ImplicitPartStmt -> ImplicitStmt -> 
! PARSE-TREE: | | DeclarationConstruct -> SpecificationConstruct -> TypeDeclarationStmt
! PARSE-TREE: | | | DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec -> 
! PARSE-TREE: | | | EntityDecl
! PARSE-TREE: | | | | Name = 'i'
! PARSE-TREE: | | | EntityDecl
! PARSE-TREE: | | | | Name = 'test_func1'
! PARSE-TREE: | | | EntityDecl
! PARSE-TREE: | | | | Name = 'test_func2'
! PARSE-TREE: | ExecutionPart -> Block
! PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> CallStmt = 'CALL test_subroutine()'
! PARSE-TREE: | | | Call
! PARSE-TREE: | | | | ProcedureDesignator -> Name = 'test_subroutine'
! PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> CompilerDirective -> InlineAlways -> 
! PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> CallStmt = 'CALL test_subroutine2()'
! PARSE-TREE: | | | Call
! PARSE-TREE: | | | | ProcedureDesignator -> Name = 'test_subroutine2'
! PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'i=test_func1()'
! PARSE-TREE: | | | Variable = 'i'
! PARSE-TREE: | | | | Designator -> DataRef -> Name = 'i'
! PARSE-TREE: | | | Expr = 'test_func1()'
! PARSE-TREE: | | | | FunctionReference -> Call
! PARSE-TREE: | | | | | ProcedureDesignator -> Name = 'test_func1'
! PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> CompilerDirective -> InlineAlways -> 
! PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'i=test_func2()'
! PARSE-TREE: | | | Variable = 'i'
! PARSE-TREE: | | | | Designator -> DataRef -> Name = 'i'
! PARSE-TREE: | | | Expr = 'test_func2()'
! PARSE-TREE: | | | | FunctionReference -> Call
! PARSE-TREE: | | | | | ProcedureDesignator -> Name = 'test_func2'
! PARSE-TREE: | EndSubroutineStmt -> 
