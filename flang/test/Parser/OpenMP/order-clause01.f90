! RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=50 %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=50 %s | FileCheck --check-prefix="PARSE-TREE" %s

! Check for ORDER([order-modifier :]concurrent) clause on OpenMP constructs

subroutine test_do_order()
 integer :: i, j = 1
 !CHECK: !$omp do order(concurrent)
 !$omp do order(concurrent)
 do i=1,10
  j = j + 1
 end do
 !$omp end do
end subroutine

!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE-NEXT: OmpBeginLoopDirective
!PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = do
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Order -> OmpOrderClause
!PARSE-TREE-NEXT: Ordering = Concurrent

subroutine test_simd_order_reproducible()
 integer :: i, j = 1
 !CHECK: !$omp simd order(reproducible:concurrent)
 !$omp simd order(reproducible:concurrent)
 do i=1,10
  j = j + 1
 end do
 !$omp end simd
end subroutine

!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE-NEXT: OmpBeginLoopDirective
!PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = simd
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Order -> OmpOrderClause
!PARSE-TREE-NEXT: OmpOrderModifier -> Value = Reproducible
!PARSE-TREE-NEXT: Ordering = Concurrent

subroutine test_do_simd_order_unconstrained()
 integer :: i, j = 1
 !CHECK: !$omp do simd order(unconstrained:concurrent)
 !$omp do simd order(unconstrained:concurrent)
 do i=1,10
  j = j + 1
 end do
 !$omp end do simd
end subroutine

!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE-NEXT: OmpBeginLoopDirective
!PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = do simd
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Order -> OmpOrderClause
!PARSE-TREE-NEXT: OmpOrderModifier -> Value = Unconstrained
!PARSE-TREE-NEXT: Ordering = Concurrent

subroutine test_parallel_do_order()
 integer :: i, j = 1
 !CHECK: !$omp parallel do order(concurrent)
 !$omp parallel do order(concurrent)
 do i=1,10
  j = j + 1
 end do
 !$omp end parallel do
end subroutine

!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE-NEXT: OmpBeginLoopDirective
!PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = parallel do
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Order -> OmpOrderClause
!PARSE-TREE-NEXT: Ordering = Concurrent

subroutine test_parallel_do_simd_order_reproducible()
 integer :: i, j = 1
 !CHECK: !$omp parallel do simd order(reproducible:concurrent)
 !$omp parallel do simd order(reproducible:concurrent)
 do i=1,10
  j = j + 1
 end do
 !$omp end parallel do simd
end subroutine

!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE-NEXT: OmpBeginLoopDirective
!PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = parallel do simd
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Order -> OmpOrderClause
!PARSE-TREE-NEXT: OmpOrderModifier -> Value = Reproducible
!PARSE-TREE-NEXT: Ordering = Concurrent

subroutine test_target_simd_order_unconstrained()
 integer :: i, j = 1
 !CHECK: !$omp target simd order(unconstrained:concurrent)
 !$omp target simd order(unconstrained:concurrent)
 do i=1,10
  j = j + 1
 end do
 !$omp end target simd
end subroutine

!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE-NEXT: OmpBeginLoopDirective
!PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = target simd
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Order -> OmpOrderClause
!PARSE-TREE-NEXT: OmpOrderModifier -> Value = Unconstrained
!PARSE-TREE-NEXT: Ordering = Concurrent

subroutine test_target_parallel_do_order()
 integer :: i, j = 1
 !CHECK: !$omp target parallel do order(concurrent)
 !$omp target parallel do order(concurrent)
 do i=1,10
  j = j + 1
 end do
 !$omp end target parallel do
end subroutine

!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE-NEXT: OmpBeginLoopDirective
!PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = target parallel do
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Order -> OmpOrderClause
!PARSE-TREE-NEXT: Ordering = Concurrent

subroutine test_target_parallel_do_simd_order_reproducible()
 integer :: i, j = 1
 !CHECK: !$omp target parallel do simd order(reproducible:concurrent)
 !$omp target parallel do simd order(reproducible:concurrent)
 do i=1,10
  j = j + 1
 end do
 !$omp end target parallel do simd
end subroutine

!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE-NEXT: OmpBeginLoopDirective
!PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = target parallel do simd
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Order -> OmpOrderClause
!PARSE-TREE-NEXT: OmpOrderModifier -> Value = Reproducible
!PARSE-TREE-NEXT: Ordering = Concurrent

subroutine test_teams_distribute_simd_order_unconstrained()
 integer :: i, j = 1
 !CHECK: !$omp teams distribute simd order(unconstrained:concurrent)
 !$omp teams distribute simd order(unconstrained:concurrent)
 do i=1,10
  j = j + 1
 end do
 !$omp end teams distribute simd
end subroutine

!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE-NEXT: OmpBeginLoopDirective
!PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = teams distribute simd
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Order -> OmpOrderClause
!PARSE-TREE-NEXT: OmpOrderModifier -> Value = Unconstrained
!PARSE-TREE-NEXT: Ordering = Concurrent

subroutine test_teams_distribute_parallel_do_order()
 integer :: i, j = 1
 !CHECK: !$omp teams distribute parallel do order(concurrent)
 !$omp teams distribute parallel do order(concurrent)
 do i=1,10
  j = j + 1
 end do
 !$omp end teams distribute parallel do
end subroutine

!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE-NEXT: OmpBeginLoopDirective
!PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = teams distribute parallel do
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Order -> OmpOrderClause
!PARSE-TREE-NEXT: Ordering = Concurrent

subroutine test_teams_distribute_parallel_do_simd_order_reproducible()
 integer :: i, j = 1
 !CHECK: !$omp teams distribute parallel do simd order(reproducible:concurrent)
 !$omp teams distribute parallel do simd order(reproducible:concurrent)
 do i=1,10
  j = j + 1
 end do
 !$omp end teams distribute parallel do simd
end subroutine

!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE-NEXT: OmpBeginLoopDirective
!PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = teams distribute parallel do simd
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Order -> OmpOrderClause
!PARSE-TREE-NEXT: OmpOrderModifier -> Value = Reproducible
!PARSE-TREE-NEXT: Ordering = Concurrent

subroutine test_target_teams_distribute_simd_order_unconstrained()
 integer :: i, j = 1
 !CHECK: !$omp target teams distribute simd order(unconstrained:concurrent)
 !$omp target teams distribute simd order(unconstrained:concurrent)
 do i=1,10
  j = j + 1
 end do
 !$omp end target teams distribute simd
end subroutine

!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE-NEXT: OmpBeginLoopDirective
!PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = target teams distribute simd
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Order -> OmpOrderClause
!PARSE-TREE-NEXT: OmpOrderModifier -> Value = Unconstrained
!PARSE-TREE-NEXT: Ordering = Concurrent

subroutine test_target_teams_distribute_parallel_do_order()
 integer :: i, j = 1
 !CHECK: !$omp target teams distribute parallel do order(concurrent)
 !$omp target teams distribute parallel do order(concurrent)
 do i=1,10
  j = j + 1
 end do
 !$omp end target teams distribute parallel do
end subroutine

!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE-NEXT: OmpBeginLoopDirective
!PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = target teams distribute parallel do
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Order -> OmpOrderClause
!PARSE-TREE-NEXT: Ordering = Concurrent

subroutine test_target_teams_distribute_parallel_do_simd_order_reproducible()
 integer :: i, j = 1
 !CHECK: !$omp target teams distribute parallel do simd order(reproducible:concurrent)
 !$omp target teams distribute parallel do simd order(reproducible:concurrent)
 do i=1,10
  j = j + 1
 end do
 !$omp end target teams distribute parallel do simd
end subroutine

!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE-NEXT: OmpBeginLoopDirective
!PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = target teams distribute parallel do simd
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Order -> OmpOrderClause
!PARSE-TREE-NEXT: OmpOrderModifier -> Value = Reproducible
!PARSE-TREE-NEXT: Ordering = Concurrent

subroutine test_taskloop_simd_order_unconstrained()
 integer :: i, j = 1
 !CHECK: !$omp taskloop simd order(unconstrained:concurrent)
 !$omp taskloop simd order(unconstrained:concurrent)
 do i=1,10
  j = j + 1
 end do
 !$omp end taskloop simd
end subroutine

!PARSE-TREE: OpenMPConstruct -> OpenMPLoopConstruct
!PARSE-TREE-NEXT: OmpBeginLoopDirective
!PARSE-TREE-NEXT: OmpLoopDirective -> llvm::omp::Directive = taskloop simd
!PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Order -> OmpOrderClause
!PARSE-TREE-NEXT: OmpOrderModifier -> Value = Unconstrained
!PARSE-TREE-NEXT: Ordering = Concurrent
