! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s 
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine test_interop_01()
  !$omp interop device(1)
  print *,'pass'
end subroutine test_interop_01

!UNPARSE: SUBROUTINE test_interop_01
!UNPARSE: !$OMP INTEROP  DEVICE(1_4)
!UNPARSE:  PRINT *, "pass"
!UNPARSE: END SUBROUTINE test_interop_01

!PARSE-TREE: | SubroutineStmt
!PARSE-TREE: | | Name = 'test_interop_01'
!PARSE-TREE: | SpecificationPart
!PARSE-TREE: | | ImplicitPart -> 
!PARSE-TREE: | ExecutionPart -> Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPInteropConstruct
!PARSE-TREE: | | | Verbatim
!PARSE-TREE: | | | OmpClauseList -> OmpClause -> Device -> OmpDeviceClause
!PARSE-TREE: | | | | Scalar -> Integer -> Expr -> LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> PrintStmt
!PARSE-TREE: | | | Format -> Star
!PARSE-TREE: | | | OutputItem -> Expr -> LiteralConstant -> CharLiteralConstant
!PARSE-TREE: | | | | string = 'pass'
!PARSE-TREE: | EndSubroutineStmt -> Name = 'test_interop_01'

subroutine test_interop_02()
  use omp_lib
  integer(omp_interop_kind) :: obj1, obj2, obj3
  !$omp interop init(targetsync: obj) use(obj1) destroy(obj3)
  print *,'pass'
end subroutine test_interop_02

!UNPARSE: SUBROUTINE test_interop_02
!UNPARSE:  USE :: omp_lib
!UNPARSE:  INTEGER(KIND=8_4) obj1, obj2, obj3
!UNPARSE: !$OMP INTEROP INIT(TARGETSYNC: obj) USE(obj1) DESTROY(obj3)
!UNPARSE:  PRINT *, "pass"
!UNPARSE: END SUBROUTINE test_interop_02

!PARSE-TREE: | SubroutineStmt
!PARSE-TREE: | | Name = 'test_interop_02'
!PARSE-TREE: | SpecificationPart
!PARSE-TREE: | | UseStmt
!PARSE-TREE: | | | Name = 'omp_lib'
!PARSE-TREE: | | ImplicitPart -> 
!PARSE-TREE: | | DeclarationConstruct -> SpecificationConstruct -> TypeDeclarationStmt
!PARSE-TREE: | | | DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec -> KindSelector -> Scalar -> Integer -> Constant -> Expr -> Designator -> DataRef -> Name = 'omp_interop_kind'
!PARSE-TREE: | | | EntityDecl
!PARSE-TREE: | | | | Name = 'obj1'
!PARSE-TREE: | | | EntityDecl
!PARSE-TREE: | | | | Name = 'obj2'
!PARSE-TREE: | | | EntityDecl
!PARSE-TREE: | | | | Name = 'obj3'
!PARSE-TREE: | ExecutionPart -> Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPInteropConstruct
!PARSE-TREE: | | | Verbatim
!PARSE-TREE: | | | OmpClauseList -> OmpClause -> Init -> OmpInitClause
!PARSE-TREE: | | | | InteropTypes -> InteropType -> Kind = TargetSync
!PARSE-TREE: | | | | InteropVar -> OmpObject -> Designator -> DataRef -> Name = 'obj'
!PARSE-TREE: | | | OmpClause -> Use -> OmpUseClause -> OmpObject -> Designator -> DataRef -> Name = 'obj1'
!PARSE-TREE: | | | OmpClause -> Destroy -> OmpDestroyClause -> OmpObject -> Designator -> DataRef -> Name = 'obj3'
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> PrintStmt
!PARSE-TREE: | | | Format -> Star
!PARSE-TREE: | | | OutputItem -> Expr -> LiteralConstant -> CharLiteralConstant
!PARSE-TREE: | | | | string = 'pass'
!PARSE-TREE: | EndSubroutineStmt -> Name = 'test_interop_02'

subroutine test_interop_03()
  use omp_lib
  Integer(omp_interop_kind) :: obj
  !$omp interop init(targetsync: obj) depend(inout: obj)
  print *,'pass'
end subroutine test_interop_03

!UNPARSE: SUBROUTINE test_interop_03
!UNPARSE:  USE :: omp_lib
!UNPARSE:  INTEGER(KIND=8_4) obj
!UNPARSE: !$OMP INTEROP INIT(TARGETSYNC: obj) DEPEND(INOUT: obj)
!UNPARSE:  PRINT *, "pass"
!UNPARSE: END SUBROUTINE test_interop_03

!PARSE-TREE: | SubroutineStmt
!PARSE-TREE: | | Name = 'test_interop_03'
!PARSE-TREE: | SpecificationPart
!PARSE-TREE: | | UseStmt
!PARSE-TREE: | | | Name = 'omp_lib'
!PARSE-TREE: | | ImplicitPart -> 
!PARSE-TREE: | | DeclarationConstruct -> SpecificationConstruct -> TypeDeclarationStmt
!PARSE-TREE: | | | DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec -> KindSelector -> Scalar -> Integer -> Constant -> Expr -> Designator -> DataRef -> Name = 'omp_interop_kind'
!PARSE-TREE: | | | EntityDecl
!PARSE-TREE: | | | | Name = 'obj'
!PARSE-TREE: | ExecutionPart -> Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPInteropConstruct
!PARSE-TREE: | | | Verbatim
!PARSE-TREE: | | | OmpClauseList -> OmpClause -> Init -> OmpInitClause
!PARSE-TREE: | | | | InteropTypes -> InteropType -> Kind = TargetSync
!PARSE-TREE: | | | | InteropVar -> OmpObject -> Designator -> DataRef -> Name = 'obj'
!PARSE-TREE: | | | OmpClause -> Depend -> OmpDependClause -> TaskDep
!PARSE-TREE: | | | | Modifier -> OmpTaskDependenceType -> Value = Inout
!PARSE-TREE: | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'obj'
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> PrintStmt
!PARSE-TREE: | | | Format -> Star
!PARSE-TREE: | | | OutputItem -> Expr -> LiteralConstant -> CharLiteralConstant
!PARSE-TREE: | | | | string = 'pass'
!PARSE-TREE: | EndSubroutineStmt -> Name = 'test_interop_03'

subroutine test_interop_04()
  use omp_lib
  integer(omp_interop_kind) :: obj
  integer, dimension(1,10) :: arr
  !$omp interop init(prefer_type("cuda"),targetsync,target: obj) depend(inout: arr) nowait
  print *,'pass'
end subroutine test_interop_04

!UNPARSE: SUBROUTINE test_interop_04
!UNPARSE:  USE :: omp_lib
!UNPARSE:  INTEGER(KIND=8_4) obj
!UNPARSE:  INTEGER, DIMENSION(1_4,10_4) :: arr
!UNPARSE: !$OMP INTEROP INIT(PREFER_TYPE("cuda"),TARGETSYNC,TARGET: obj) DEPEND(INOUT: a&
!UNPARSE: !$OMP&rr) NOWAIT
!UNPARSE:  PRINT *, "pass"
!UNPARSE: END SUBROUTINE test_interop_04

!PARSE-TREE: | SubroutineStmt
!PARSE-TREE: | | Name = 'test_interop_04'
!PARSE-TREE: | SpecificationPart
!PARSE-TREE: | | UseStmt
!PARSE-TREE: | | | Name = 'omp_lib'
!PARSE-TREE: | | ImplicitPart -> 
!PARSE-TREE: | | DeclarationConstruct -> SpecificationConstruct -> TypeDeclarationStmt
!PARSE-TREE: | | | DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec -> KindSelector -> Scalar -> Integer -> Constant -> Expr -> Designator -> DataRef -> Name = 'omp_interop_kind'
!PARSE-TREE: | | | EntityDecl
!PARSE-TREE: | | | | Name = 'obj'
!PARSE-TREE: | | DeclarationConstruct -> SpecificationConstruct -> TypeDeclarationStmt
!PARSE-TREE: | | | DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec -> 
!PARSE-TREE: | | | AttrSpec -> ArraySpec -> ExplicitShapeSpec
!PARSE-TREE: | | | | SpecificationExpr -> Scalar -> Integer -> Expr -> LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | ExplicitShapeSpec
!PARSE-TREE: | | | | SpecificationExpr -> Scalar -> Integer -> Expr -> LiteralConstant -> IntLiteralConstant = '10'
!PARSE-TREE: | | | EntityDecl
!PARSE-TREE: | | | | Name = 'arr'
!PARSE-TREE: | ExecutionPart -> Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPInteropConstruct
!PARSE-TREE: | | | Verbatim
!PARSE-TREE: | | | OmpClauseList -> OmpClause -> Init -> OmpInitClause
!PARSE-TREE: | | | | InteropModifier -> InteropPreference -> CharLiteralConstant
!PARSE-TREE: | | | | | string = 'cuda'
!PARSE-TREE: | | | | InteropTypes -> InteropType -> Kind = TargetSync
!PARSE-TREE: | | | | InteropType -> Kind = Target
!PARSE-TREE: | | | | InteropVar -> OmpObject -> Designator -> DataRef -> Name = 'obj'
!PARSE-TREE: | | | OmpClause -> Depend -> OmpDependClause -> TaskDep
!PARSE-TREE: | | | | Modifier -> OmpTaskDependenceType -> Value = Inout
!PARSE-TREE: | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'arr'
!PARSE-TREE: | | | OmpClause -> Nowait
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> PrintStmt
!PARSE-TREE: | | | Format -> Star
!PARSE-TREE: | | | OutputItem -> Expr -> LiteralConstant -> CharLiteralConstant
!PARSE-TREE: | | | | string = 'pass'
!PARSE-TREE: | EndSubroutineStmt -> Name = 'test_interop_04'


subroutine test_interop_05()
  use omp_lib
  integer(omp_interop_kind) :: obj
  !$omp interop init(prefer_type(omp_ifr_sycl), targetsync: obj) device(device_num:0)
  print *,'pass'
end subroutine test_interop_05

!UNPARSE: SUBROUTINE test_interop_05
!UNPARSE:  USE :: omp_lib
!UNPARSE:  INTEGER(KIND=8_4) obj
!UNPARSE: !$OMP INTEROP INIT(PREFER_TYPE(4_4),TARGETSYNC: obj) DEVICE(DEVICE_NUM: 0_4)
!UNPARSE:  PRINT *, "pass"
!UNPARSE: END SUBROUTINE test_interop_05

!PARSE-TREE: | SubroutineStmt
!PARSE-TREE: | | Name = 'test_interop_05'
!PARSE-TREE: | SpecificationPart
!PARSE-TREE: | | UseStmt
!PARSE-TREE: | | | Name = 'omp_lib'
!PARSE-TREE: | | ImplicitPart -> 
!PARSE-TREE: | | DeclarationConstruct -> SpecificationConstruct -> TypeDeclarationStmt
!PARSE-TREE: | | | DeclarationTypeSpec -> IntrinsicTypeSpec -> IntegerTypeSpec -> KindSelector -> Scalar -> Integer -> Constant -> Expr -> Designator -> DataRef -> Name = 'omp_interop_kind'
!PARSE-TREE: | | | EntityDecl
!PARSE-TREE: | | | | Name = 'obj'
!PARSE-TREE: | ExecutionPart -> Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPInteropConstruct
!PARSE-TREE: | | | Verbatim
!PARSE-TREE: | | | OmpClauseList -> OmpClause -> Init -> OmpInitClause
!PARSE-TREE: | | | | InteropModifier -> InteropPreference -> Scalar -> Integer -> Constant -> Expr -> Designator -> DataRef -> Name = 'omp_ifr_sycl'
!PARSE-TREE: | | | | InteropTypes -> InteropType -> Kind = TargetSync
!PARSE-TREE: | | | | InteropVar -> OmpObject -> Designator -> DataRef -> Name = 'obj'
!PARSE-TREE: | | | OmpClause -> Device -> OmpDeviceClause
!PARSE-TREE: | | | | Modifier -> OmpDeviceModifier -> Value = Device_Num
!PARSE-TREE: | | | | Scalar -> Integer -> Expr -> LiteralConstant -> IntLiteralConstant = '0'
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> PrintStmt
!PARSE-TREE: | | | Format -> Star
!PARSE-TREE: | | | OutputItem -> Expr -> LiteralConstant -> CharLiteralConstant
!PARSE-TREE: | | | | string = 'pass'
!PARSE-TREE: | EndSubroutineStmt -> Name = 'test_interop_05'

