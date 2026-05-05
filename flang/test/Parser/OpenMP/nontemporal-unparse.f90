!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=50 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=50 %s | FileCheck --check-prefix="PARSE-TREE" %s

program omp_simd
  integer i
  integer, allocatable :: a(:)

  allocate(a(10))

  !$omp simd nontemporal(a)
  do i = 1, 10
    a(i) = i
  end do
  !$omp end simd
end program omp_simd

!UNPARSE: PROGRAM OMP_SIMD
!UNPARSE:  INTEGER i
!UNPARSE:  INTEGER, ALLOCATABLE :: a(:)
!UNPARSE:  ALLOCATE(a(10_4))
!UNPARSE: !$OMP SIMD NONTEMPORAL(a)
!UNPARSE:  DO i=1_4,10_4
!UNPARSE:    a(int(i,kind=8))=i
!UNPARSE:  END DO
!UNPARSE: !$OMP END SIMD
!UNPARSE: END PROGRAM OMP_SIMD

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE: | OmpBeginDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = simd
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Nontemporal -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'a'
!PARSE-TREE: | | Flags = {}
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
!PARSE-TREE: | | | NonLabelDoStmt
!PARSE-TREE: | | | | LoopControl -> LoopBounds
!PARSE-TREE: | | | | | Scalar -> Name = 'i'
!PARSE-TREE: | | | | | Scalar -> Expr = '1_4'
!PARSE-TREE: | | | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | | | Scalar -> Expr = '10_4'
!PARSE-TREE: | | | | | | LiteralConstant -> IntLiteralConstant = '10'
!PARSE-TREE: | | | Block
!PARSE-TREE: | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'a(int(i,kind=8))=i'
!PARSE-TREE: | | | | | Variable = 'a(int(i,kind=8))'
!PARSE-TREE: | | | | | | Designator -> DataRef -> ArrayElement
!PARSE-TREE: | | | | | | | DataRef -> Name = 'a'
!PARSE-TREE: | | | | | | | SectionSubscript -> Integer -> Expr = 'i'
!PARSE-TREE: | | | | | | | | Designator -> DataRef -> Name = 'i'
!PARSE-TREE: | | | | | Expr = 'i'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'i'
!PARSE-TREE: | | | EndDoStmt ->
!PARSE-TREE: | OmpEndDirective
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = simd
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = {}
