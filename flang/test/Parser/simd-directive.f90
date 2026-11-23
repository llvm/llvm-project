! Test parsing and unparsing of SIMD Directive

! RUN: %flang_fc1 -fdebug-dump-parse-tree %s -o - | FileCheck %s --check-prefix="PARSE-TREE"
! RUN: %flang_fc1 -fdebug-unparse %s -o - | FileCheck %s --check-prefix="UNPARSE"

subroutine test()
  integer :: i, n, sum
  n = 10
  sum = 0

  !DIR$ SIMD
  do i = 1, n
    sum = sum + i
  end do
end subroutine

! UNPARSE: SUBROUTINE test
! UNPARSE:  INTEGER i, n, sum
! UNPARSE:   n=10_4
! UNPARSE:   sum=0_4
! UNPARSE:  !DIR$ SIMD
! UNPARSE:  DO i=1_4,n
! UNPARSE:    sum=sum+i
! UNPARSE:  END DO
! UNPARSE: END SUBROUTINE

! PARSE-TREE: ======================== Flang: parse tree dump ========================
! PARSE-TREE: Program -> ProgramUnit -> SubroutineSubprogram
! PARSE-TREE: | SubroutineStmt
! PARSE-TREE: | | Name = 'test'
! PARSE-TREE: | SpecificationPart
! PARSE-TREE: | | ImplicitPart -> 
! PARSE-TREE: | | DeclarationConstruct -> SpecificationConstruct -> TypeDeclarationStmt
! PARSE-TREE: | | | DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec -> 
! PARSE-TREE: | | | EntityDecl
! PARSE-TREE: | | | | Name = 'i'
! PARSE-TREE: | | | EntityDecl
! PARSE-TREE: | | | | Name = 'n'
! PARSE-TREE: | | | EntityDecl
! PARSE-TREE: | | | | Name = 'sum'
! PARSE-TREE: | ExecutionPart -> Block
! PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'n=10_4'
! PARSE-TREE: | | | Variable = 'n'
! PARSE-TREE: | | | | Designator -> DataRef -> Name = 'n'
! PARSE-TREE: | | | Expr = '10_4'
! PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '10'
! PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'sum=0_4'
! PARSE-TREE: | | | Variable = 'sum'
! PARSE-TREE: | | | | Designator -> DataRef -> Name = 'sum'
! PARSE-TREE: | | | Expr = '0_4'
! PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '0'
! PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> CompilerDirective -> Simd
! PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
! PARSE-TREE: | | | NonLabelDoStmt
! PARSE-TREE: | | | | LoopControl -> LoopBounds
! PARSE-TREE: | | | | | Scalar -> Name = 'i'
! PARSE-TREE: | | | | | Scalar -> Expr = '1_4'
! PARSE-TREE: | | | | | | LiteralConstant -> IntLiteralConstant = '1'
! PARSE-TREE: | | | | | Scalar -> Expr = 'n'
! PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'n'
! PARSE-TREE: | | | Block
! PARSE-TREE: | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'sum=sum+i'
! PARSE-TREE: | | | | | Variable = 'sum'
! PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'sum'
! PARSE-TREE: | | | | | Expr = 'sum+i'
! PARSE-TREE: | | | | | | Add
! PARSE-TREE: | | | | | | | Expr = 'sum'
! PARSE-TREE: | | | | | | | | Designator -> DataRef -> Name = 'sum'
! PARSE-TREE: | | | | | | | Expr = 'i'
! PARSE-TREE: | | | | | | | | Designator -> DataRef -> Name = 'i'
! PARSE-TREE: | | | EndDoStmt -> 
! PARSE-TREE: | EndSubroutineStmt ->
