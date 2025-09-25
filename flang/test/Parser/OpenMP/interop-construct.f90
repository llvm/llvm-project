! REQUIRES: openmp_runtime
! RUN: %flang_fc1 -fdebug-unparse -fopenmp-version=52 %openmp_flags %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s 
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema -fopenmp-version=52 %openmp_flags %s | FileCheck --check-prefix="PARSE-TREE" %s

SUBROUTINE test_interop_01()
  !$OMP INTEROP DEVICE(1)
  PRINT *,'pass'
END SUBROUTINE test_interop_01

!UNPARSE: SUBROUTINE test_interop_01
!UNPARSE: !$OMP INTEROP  DEVICE(1_4)
!UNPARSE:  PRINT *, "pass"
!UNPARSE: END SUBROUTINE test_interop_01

!PARSE-TREE: | SubroutineStmt
!PARSE-TREE: | | Name = 'test_interop_01'
!PARSE-TREE: | SpecificationPart
!PARSE-TREE: | | ImplicitPart -> 
!PARSE-TREE: | ExecutionPart -> Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPInteropConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = interop
!PARSE-TREE: | | | OmpClauseList -> OmpClause -> Device -> OmpDeviceClause
!PARSE-TREE: | | | | Scalar -> Integer -> Expr -> LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | Flags = None
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> PrintStmt
!PARSE-TREE: | | | Format -> Star
!PARSE-TREE: | | | OutputItem -> Expr -> LiteralConstant -> CharLiteralConstant
!PARSE-TREE: | | | | string = 'pass'
!PARSE-TREE: | EndSubroutineStmt -> Name = 'test_interop_01'

SUBROUTINE test_interop_02()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: obj1, obj2, obj3
  !$OMP INTEROP INIT(TARGETSYNC: obj) USE(obj1) DESTROY(obj3)
  PRINT *,'pass'
END SUBROUTINE test_interop_02

!UNPARSE: SUBROUTINE test_interop_02
!UNPARSE:  USE :: omp_lib
!UNPARSE:  INTEGER(KIND=8_4) obj1, obj2, obj3
!UNPARSE: !$OMP INTEROP  INIT(TARGETSYNC: obj) USE(obj1) DESTROY(obj3)
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
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPInteropConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = interop
!PARSE-TREE: | | | OmpClauseList -> OmpClause -> Init -> OmpInitClause
!PARSE-TREE: | | | | Modifier -> OmpInteropType -> Value = TargetSync
!PARSE-TREE: | | | | OmpObject -> Designator -> DataRef -> Name = 'obj'
!PARSE-TREE: | | | OmpClause -> Use -> OmpUseClause -> OmpObject -> Designator -> DataRef -> Name = 'obj1'
!PARSE-TREE: | | | OmpClause -> Destroy -> OmpDestroyClause -> OmpObject -> Designator -> DataRef -> Name = 'obj3'
!PARSE-TREE: | | | Flags = None
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> PrintStmt
!PARSE-TREE: | | | Format -> Star
!PARSE-TREE: | | | OutputItem -> Expr -> LiteralConstant -> CharLiteralConstant
!PARSE-TREE: | | | | string = 'pass'
!PARSE-TREE: | EndSubroutineStmt -> Name = 'test_interop_02'

SUBROUTINE test_interop_03()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: obj
  !$OMP INTEROP INIT(TARGETSYNC: obj) DEPEND(INOUT: obj)
  PRINT *,'pass'
END SUBROUTINE test_interop_03

!UNPARSE: SUBROUTINE test_interop_03
!UNPARSE:  USE :: omp_lib
!UNPARSE:  INTEGER(KIND=8_4) obj
!UNPARSE: !$OMP INTEROP  INIT(TARGETSYNC: obj) DEPEND(INOUT: obj)
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
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPInteropConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = interop
!PARSE-TREE: | | | OmpClauseList -> OmpClause -> Init -> OmpInitClause
!PARSE-TREE: | | | | Modifier -> OmpInteropType -> Value = TargetSync
!PARSE-TREE: | | | | OmpObject -> Designator -> DataRef -> Name = 'obj'
!PARSE-TREE: | | | OmpClause -> Depend -> OmpDependClause -> TaskDep
!PARSE-TREE: | | | | Modifier -> OmpTaskDependenceType -> Value = Inout
!PARSE-TREE: | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'obj'
!PARSE-TREE: | | | Flags = None
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> PrintStmt
!PARSE-TREE: | | | Format -> Star
!PARSE-TREE: | | | OutputItem -> Expr -> LiteralConstant -> CharLiteralConstant
!PARSE-TREE: | | | | string = 'pass'
!PARSE-TREE: | EndSubroutineStmt -> Name = 'test_interop_03'

SUBROUTINE test_interop_04()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: obj
  INTEGER, DIMENSION(1,10) :: arr
  !$OMP INTEROP INIT(PREFER_TYPE("cuda"),TARGETSYNC,TARGET: obj) DEPEND(INOUT: arr) NOWAIT
  PRINT *,'pass'
END SUBROUTINE test_interop_04

!UNPARSE: SUBROUTINE test_interop_04
!UNPARSE:  USE :: omp_lib
!UNPARSE:  INTEGER(KIND=8_4) obj
!UNPARSE:  INTEGER, DIMENSION(1_4,10_4) :: arr
!UNPARSE: !$OMP INTEROP  INIT(PREFER_TYPE("cuda"),TARGETSYNC,TARGET: obj) DEPEND(INOUT: &
!UNPARSE: !$OMP&arr) NOWAIT
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
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPInteropConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = interop
!PARSE-TREE: | | | OmpClauseList -> OmpClause -> Init -> OmpInitClause
!PARSE-TREE: | | | | Modifier -> OmpInteropPreference -> OmpInteropRuntimeIdentifier -> CharLiteralConstant
!PARSE-TREE: | | | | | string = 'cuda'
!PARSE-TREE: | | | | Modifier -> OmpInteropType -> Value = TargetSync
!PARSE-TREE: | | | | Modifier -> OmpInteropType -> Value = Target
!PARSE-TREE: | | | | OmpObject -> Designator -> DataRef -> Name = 'obj'
!PARSE-TREE: | | | OmpClause -> Depend -> OmpDependClause -> TaskDep
!PARSE-TREE: | | | | Modifier -> OmpTaskDependenceType -> Value = Inout
!PARSE-TREE: | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'arr'
!PARSE-TREE: | | | OmpClause -> Nowait
!PARSE-TREE: | | | Flags = None
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> PrintStmt
!PARSE-TREE: | | | Format -> Star
!PARSE-TREE: | | | OutputItem -> Expr -> LiteralConstant -> CharLiteralConstant
!PARSE-TREE: | | | | string = 'pass'
!PARSE-TREE: | EndSubroutineStmt -> Name = 'test_interop_04'

SUBROUTINE test_interop_05()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: obj
  !$OMP INTEROP INIT(PREFER_TYPE(omp_ifr_sycl), TARGETSYNC: obj) DEVICE(DEVICE_NUM:0)
  PRINT *,'pass'
END SUBROUTINE test_interop_05

!UNPARSE: SUBROUTINE test_interop_05
!UNPARSE:  USE :: omp_lib
!UNPARSE:  INTEGER(KIND=8_4) obj
!UNPARSE: !$OMP INTEROP  INIT(PREFER_TYPE(4_4),TARGETSYNC: obj) DEVICE(DEVICE_NUM: 0_4)
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
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPInteropConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | | | OmpDirectiveName -> llvm::omp::Directive = interop
!PARSE-TREE: | | | OmpClauseList -> OmpClause -> Init -> OmpInitClause
!PARSE-TREE: | | | | Modifier -> OmpInteropPreference -> OmpInteropRuntimeIdentifier -> Scalar -> Integer -> Constant -> Expr -> Designator -> DataRef -> Name = 'omp_ifr_sycl'
!PARSE-TREE: | | | | Modifier -> OmpInteropType -> Value = TargetSync
!PARSE-TREE: | | | | OmpObject -> Designator -> DataRef -> Name = 'obj'
!PARSE-TREE: | | | OmpClause -> Device -> OmpDeviceClause
!PARSE-TREE: | | | | Modifier -> OmpDeviceModifier -> Value = Device_Num
!PARSE-TREE: | | | | Scalar -> Integer -> Expr -> LiteralConstant -> IntLiteralConstant = '0'
!PARSE-TREE: | | | Flags = None
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> PrintStmt
!PARSE-TREE: | | | Format -> Star
!PARSE-TREE: | | | OutputItem -> Expr -> LiteralConstant -> CharLiteralConstant
!PARSE-TREE: | | | | string = 'pass'
!PARSE-TREE: | EndSubroutineStmt -> Name = 'test_interop_05'

