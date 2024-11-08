! RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=50 %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=50 %s | FileCheck --check-prefix="PARSE-TREE" %s

! Check for IN_REDUCTION() clause on OpenMP constructs

subroutine omp_in_reduction_taskgroup()
    integer :: z, i
    !CHECK: !$OMP TASKGROUP  TASK_REDUCTION(+:z)
    !$omp taskgroup task_reduction(+:z)
    !CHECK-NEXT: !$OMP TASK  IN_REDUCTION(+:z)
        !$omp task in_reduction(+:z)
    !CHECK-NEXT: z=z+5_4
            z = z + 5
    !CHECK-NEXT: !$OMP END TASK
        !$omp end task

    !CHECK-NEXT: !$OMP TASKLOOP  IN_REDUCTION(+:z)
        !$omp taskloop in_reduction(+:z)
    !CHECK-NEXT: DO i=1_4,10_4
            do i=1,10
    !CHECK-NEXT: z=5_4*z
                z = z * 5
    !CHECK-NEXT: END DO
            end do
    !CHECK-NEXT: !$OMP END TASKLOOP
        !$omp end taskloop
    !CHECK-NEXT: !$OMP END TASKGROUP
    !$omp end taskgroup
end subroutine omp_in_reduction_taskgroup

!PARSE-TREE: OpenMPConstruct -> OpenMPBlockConstruct
!PARSE-TREE-NEXT: OmpBeginBlockDirective
!PARSE-TREE-NEXT: OmpBlockDirective -> llvm::omp::Directive = taskgroup
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> TaskReduction -> OmpReductionClause

!PARSE-TREE: OpenMPConstruct -> OpenMPBlockConstruct
!PARSE-TREE-NEXT: OmpBeginBlockDirective
!PARSE-TREE-NEXT: OmpBlockDirective -> llvm::omp::Directive = task
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> InReduction -> OmpInReductionClause
!PARSE-TREE-NEXT: OmpReductionOperator -> DefinedOperator -> IntrinsicOperator = Add
!PARSE-TREE-NEXT: OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'z'

!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE-NEXT: OmpBeginLoopDirective
!PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = taskloop
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> InReduction -> OmpInReductionClause
!PARSE-TREE-NEXT: OmpReductionOperator -> DefinedOperator -> IntrinsicOperator = Add
!PARSE-TREE-NEXT: OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'z'

subroutine omp_in_reduction_parallel()
    integer :: z
    !CHECK: !$OMP PARALLEL  REDUCTION(+:z)
    !$omp parallel reduction(+:z)
    !CHECK-NEXT: !$OMP TASKLOOP SIMD  IN_REDUCTION(+:z)
        !$omp taskloop simd in_reduction(+:z)
    !CHECK-NEXT: DO i=1_4,10_4
            do i=1,10
    !CHECK-NEXT: z=5_4*z
                z = z * 5
    !CHECK-NEXT: END DO
            end do
    !CHECK-NEXT: !$OMP END TASKLOOP SIMD
        !$omp end taskloop simd
    !CHECK-NEXT: !$OMP END PARALLEL
    !$omp end parallel
end subroutine omp_in_reduction_parallel

!PARSE-TREE: OpenMPConstruct -> OpenMPBlockConstruct
!PARSE-TREE-NEXT: OmpBeginBlockDirective
!PARSE-TREE-NEXT: OmpBlockDirective -> llvm::omp::Directive = parallel
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Reduction -> OmpReductionClause

!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE-NEXT: OmpBeginLoopDirective
!PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = taskloop simd
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> InReduction -> OmpInReductionClause
!PARSE-TREE-NEXT: OmpReductionOperator -> DefinedOperator -> IntrinsicOperator = Add
!PASRE-TREE-NEXT: OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'z'

