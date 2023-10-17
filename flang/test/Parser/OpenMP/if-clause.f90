! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck %s

program openmp_parse_if
  logical :: cond
  integer :: i

  ! CHECK: OmpSimpleStandaloneDirective -> llvm::omp::Directive = target update
  ! CHECK-NEXT: OmpClause -> If -> OmpIfClause
  ! CHECK-NOT: DirectiveNameModifier
  !$omp target update if(cond)

  ! CHECK: OmpSimpleStandaloneDirective -> llvm::omp::Directive = target update
  ! CHECK-NEXT: OmpClause -> If -> OmpIfClause
  ! CHECK-NEXT: DirectiveNameModifier = TargetUpdate
  !$omp target update if(target update: cond)

  ! CHECK: OmpSimpleStandaloneDirective -> llvm::omp::Directive = target enter data
  ! CHECK: OmpClause -> If -> OmpIfClause
  ! CHECK-NEXT: DirectiveNameModifier = TargetEnterData
  !$omp target enter data map(to: i) if(target enter data: cond)

  ! CHECK: OmpSimpleStandaloneDirective -> llvm::omp::Directive = target exit data
  ! CHECK: OmpClause -> If -> OmpIfClause
  ! CHECK-NEXT: DirectiveNameModifier = TargetExitData
  !$omp target exit data map(from: i) if(target exit data: cond)

  ! CHECK: OmpBlockDirective -> llvm::omp::Directive = target data
  ! CHECK: OmpClause -> If -> OmpIfClause
  ! CHECK-NEXT: DirectiveNameModifier = TargetData
  !$omp target data map(tofrom: i) if(target data: cond)
  !$omp end target data

  ! CHECK: OmpLoopDirective -> llvm::omp::Directive = target teams distribute parallel do simd
  ! CHECK: OmpClause -> If -> OmpIfClause
  ! CHECK-NEXT: DirectiveNameModifier = Target
  ! CHECK: OmpClause -> If -> OmpIfClause
  ! CHECK-NEXT: DirectiveNameModifier = Teams
  ! CHECK: OmpClause -> If -> OmpIfClause
  ! CHECK-NEXT: DirectiveNameModifier = Parallel
  ! CHECK: OmpClause -> If -> OmpIfClause
  ! CHECK-NEXT: DirectiveNameModifier = Simd
  !$omp target teams distribute parallel do simd if(target: cond) &
  !$omp&    if(teams: cond) if(parallel: cond) if(simd: cond)
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do simd

  ! CHECK: OmpBlockDirective -> llvm::omp::Directive = task
  ! CHECK-NEXT: OmpClause -> If -> OmpIfClause
  ! CHECK-NEXT: DirectiveNameModifier = Task
  !$omp task if(task: cond)
  !$omp end task

  ! CHECK: OmpLoopDirective -> llvm::omp::Directive = taskloop
  ! CHECK-NEXT: OmpClause -> If -> OmpIfClause
  ! CHECK-NEXT: DirectiveNameModifier = Taskloop
  !$omp taskloop if(taskloop: cond)
  do i = 1, 10
  end do
  !$omp end taskloop
end program openmp_parse_if
