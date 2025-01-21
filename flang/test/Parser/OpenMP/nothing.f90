!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=51 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=51 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00
  !$omp nothing
end

!UNPARSE: SUBROUTINE f00
!UNPARSE:  !$OMP NOTHING
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPart -> Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPUtilityConstruct -> OmpNothingDirective

subroutine f01
  block
  import, none
  integer :: x
  !$omp nothing   ! "nothing" in the execution part
  x = x+1
  end block
end

!UNPARSE: SUBROUTINE f01
!UNPARSE:  BLOCK
!UNPARSE:   IMPORT, NONE
!UNPARSE:   INTEGER x
!UNPARSE:   !$OMP NOTHING
!UNPARSE:    x=x+1_4
!UNPARSE:  END BLOCK
!UNPARSE: END SUBROUTINE

!PARSE-TREE: BlockStmt ->
!PARSE-TREE: BlockSpecificationPart -> SpecificationPart
!PARSE-TREE: | ImportStmt
!PARSE-TREE: | ImplicitPart ->
!PARSE-TREE: | DeclarationConstruct -> SpecificationConstruct -> TypeDeclarationStmt
!PARSE-TREE: | | DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec ->
!PARSE-TREE: | | EntityDecl
!PARSE-TREE: | | | Name = 'x'
!PARSE-TREE: Block
!PARSE-TREE: | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPUtilityConstruct -> OmpNothingDirective
!PARSE-TREE: | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'x=x+1_4'
!PARSE-TREE: | | Variable = 'x'
!PARSE-TREE: | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | Expr = 'x+1_4'
!PARSE-TREE: | | | Add
!PARSE-TREE: | | | | Expr = 'x'
!PARSE-TREE: | | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | Expr = '1_4'
!PARSE-TREE: | | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: EndBlockStmt ->

subroutine f02
  integer :: x
  !$omp nothing
end

!UNPARSE: SUBROUTINE f02
!UNPARSE:  INTEGER x
!UNPARSE:  !$OMP NOTHING
!UNPARSE: END SUBROUTINE

!PARSE-TREE: SpecificationPart
!PARSE-TREE: | ImplicitPart ->
!PARSE-TREE: | DeclarationConstruct -> SpecificationConstruct -> TypeDeclarationStmt
!PARSE-TREE: | | DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec ->
!PARSE-TREE: | | EntityDecl
!PARSE-TREE: | | | Name = 'x'
!PARSE-TREE: ExecutionPart -> Block
!PARSE-TREE: | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPUtilityConstruct -> OmpNothingDirective

subroutine f03
  block
  !$omp nothing   ! "nothing" in the specification part
  import, none
  integer :: x
  x = x+1
  end block
end

!UNPARSE: SUBROUTINE f03
!UNPARSE:  BLOCK
!UNPARSE:   !$OMP NOTHING
!UNPARSE:   IMPORT, NONE
!UNPARSE:   INTEGER x
!UNPARSE:    x=x+1_4
!UNPARSE:  END BLOCK
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPart -> Block
!PARSE-TREE: | ExecutionPartConstruct -> ExecutableConstruct -> BlockConstruct
!PARSE-TREE: | | BlockStmt ->
!PARSE-TREE: | | BlockSpecificationPart -> SpecificationPart
!PARSE-TREE: | | | OpenMPDeclarativeConstruct -> OpenMPUtilityConstruct -> OmpNothingDirective
!PARSE-TREE: | | | ImportStmt
!PARSE-TREE: | | | ImplicitPart ->
!PARSE-TREE: | | | DeclarationConstruct -> SpecificationConstruct -> TypeDeclarationStmt
!PARSE-TREE: | | | | DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec ->
!PARSE-TREE: | | | | EntityDecl
!PARSE-TREE: | | | | | Name = 'x'
!PARSE-TREE: | | Block
!PARSE-TREE: | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'x=x+1_4'
!PARSE-TREE: | | | | Variable = 'x'
!PARSE-TREE: | | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | Expr = 'x+1_4'
!PARSE-TREE: | | | | | Add
!PARSE-TREE: | | | | | | Expr = 'x'
!PARSE-TREE: | | | | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | | | Expr = '1_4'
!PARSE-TREE: | | | | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | EndBlockStmt ->
!PARSE-TREE: EndSubroutineStmt ->