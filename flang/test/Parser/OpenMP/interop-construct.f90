! REQUIRES: openmp_runtime
! RUN: %flang_fc1 -fdebug-unparse -fopenmp-version=60 %openmp_flags %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema -fopenmp-version=60 %openmp_flags %s | FileCheck --check-prefix="PARSE-TREE" %s

SUBROUTINE test_interop_01()
  !$OMP INTEROP DEVICE(1)
  PRINT *,'pass'
END SUBROUTINE test_interop_01

!UNPARSE: SUBROUTINE test_interop_01
!UNPARSE: !$OMP INTEROP  DEVICE(1_4)
!UNPARSE:  PRINT *, "pass"
!UNPARSE: END SUBROUTINE test_interop_01

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPInteropConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = interop
!PARSE-TREE: | OmpClauseList -> OmpClause -> Device -> OmpDeviceClause
!PARSE-TREE: | | Scalar -> Integer -> Expr -> LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | Flags = {}


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

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPInteropConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = interop
!PARSE-TREE: | OmpClauseList -> OmpClause -> Init -> OmpInitClause
!PARSE-TREE: | | Modifier -> OmpInteropType -> Value = Targetsync
!PARSE-TREE: | | OmpObject -> Designator -> DataRef -> Name = 'obj'
!PARSE-TREE: | OmpClause -> Use -> OmpUseClause -> OmpObject -> Designator -> DataRef -> Name = 'obj1'
!PARSE-TREE: | OmpClause -> Destroy -> OmpDestroyClause -> OmpObject -> Designator -> DataRef -> Name = 'obj3'
!PARSE-TREE: | Flags = {}


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

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPInteropConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = interop
!PARSE-TREE: | OmpClauseList -> OmpClause -> Init -> OmpInitClause
!PARSE-TREE: | | Modifier -> OmpInteropType -> Value = Targetsync
!PARSE-TREE: | | OmpObject -> Designator -> DataRef -> Name = 'obj'
!PARSE-TREE: | OmpClause -> Depend -> OmpDependClause -> TaskDep
!PARSE-TREE: | | Modifier -> OmpTaskDependenceType -> Value = Inout
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'obj'
!PARSE-TREE: | Flags = {}


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
!UNPARSE: !$OMP INTEROP INIT(PREFER_TYPE("cuda"), TARGETSYNC, TARGET: obj) DEPEND(INOUT: arr) &
!UNPARSE: !$OMP&NOWAIT
!UNPARSE:  PRINT *, "pass"
!UNPARSE: END SUBROUTINE test_interop_04

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPInteropConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = interop
!PARSE-TREE: | OmpClauseList -> OmpClause -> Init -> OmpInitClause
!PARSE-TREE: | | Modifier -> OmpPreferType -> OmpPreferenceSpecification -> Expr -> LiteralConstant -> CharLiteralConstant
!PARSE-TREE: | | | string = 'cuda'
!PARSE-TREE: | | Modifier -> OmpInteropType -> Value = Targetsync
!PARSE-TREE: | | Modifier -> OmpInteropType -> Value = Target
!PARSE-TREE: | | OmpObject -> Designator -> DataRef -> Name = 'obj'
!PARSE-TREE: | OmpClause -> Depend -> OmpDependClause -> TaskDep
!PARSE-TREE: | | Modifier -> OmpTaskDependenceType -> Value = Inout
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'arr'
!PARSE-TREE: | OmpClause -> Nowait
!PARSE-TREE: | Flags = {}


SUBROUTINE test_interop_05()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: obj
  !$OMP INTEROP INIT(PREFER_TYPE(omp_ifr_sycl), TARGETSYNC: obj) DEVICE(DEVICE_NUM:0)
  PRINT *,'pass'
END SUBROUTINE test_interop_05

!UNPARSE: SUBROUTINE test_interop_05
!UNPARSE:  USE :: omp_lib
!UNPARSE:  INTEGER(KIND=8_4) obj
!UNPARSE: !$OMP INTEROP INIT(PREFER_TYPE(4_4), TARGETSYNC: obj) DEVICE(DEVICE_NUM: 0_4)
!UNPARSE:  PRINT *, "pass"
!UNPARSE: END SUBROUTINE test_interop_05

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPInteropConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = interop
!PARSE-TREE: | OmpClauseList -> OmpClause -> Init -> OmpInitClause
!PARSE-TREE: | | Modifier -> OmpPreferType -> OmpPreferenceSpecification -> Expr -> Designator -> DataRef -> Name = 'omp_ifr_sycl'
!PARSE-TREE: | | Modifier -> OmpInteropType -> Value = Targetsync
!PARSE-TREE: | | OmpObject -> Designator -> DataRef -> Name = 'obj'
!PARSE-TREE: | OmpClause -> Device -> OmpDeviceClause
!PARSE-TREE: | | Modifier -> OmpDeviceModifier -> Value = Device_Num
!PARSE-TREE: | | Scalar -> Integer -> Expr -> LiteralConstant -> IntLiteralConstant = '0'
!PARSE-TREE: | Flags = {}


SUBROUTINE test_interop_06()
  USE omp_lib
  INTEGER(KIND=OMP_INTEROP_KIND) :: obj
  !$OMP INTEROP INIT(PREFER_TYPE({FR("some_runtime"), ATTR("ext1", "ext2")}), TARGETSYNC: obj)
  PRINT *, 'pass'
END

!UNPARSE: SUBROUTINE test_interop_06
!UNPARSE:  USE :: omp_lib
!UNPARSE:  INTEGER(KIND=8_4) obj
!UNPARSE: !$OMP INTEROP INIT(PREFER_TYPE({FR("some_runtime"), ATTR("ext1", "ext2")}), TARGETSYNC: obj)
!UNPARSE:  PRINT *, "pass"
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPInteropConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = interop
!PARSE-TREE: | OmpClauseList -> OmpClause -> Init -> OmpInitClause
!PARSE-TREE: | | Modifier -> OmpPreferType -> OmpPreferenceSpecification -> OmpPreferenceSelector -> Expr -> LiteralConstant -> CharLiteralConstant
!PARSE-TREE: | | | string = 'some_runtime'
!PARSE-TREE: | | OmpPreferenceSelector -> Expr -> LiteralConstant -> CharLiteralConstant
!PARSE-TREE: | | | string = 'ext1'
!PARSE-TREE: | | Expr -> LiteralConstant -> CharLiteralConstant
!PARSE-TREE: | | | string = 'ext2'
!PARSE-TREE: | | Modifier -> OmpInteropType -> Value = Targetsync
!PARSE-TREE: | | OmpObject -> Designator -> DataRef -> Name = 'obj'
!PARSE-TREE: | Flags = {}
