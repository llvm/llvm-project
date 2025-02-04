! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=52 %s | FileCheck %s

program openmp_parse_if
  logical :: cond
  integer :: i

  ! CHECK: OmpSimpleStandaloneDirective -> llvm::omp::Directive = target update
  ! CHECK-NEXT: OmpClause -> If -> OmpIfClause
  ! CHECK-NOT: DirectiveNameModifier
  !$omp target update if(cond) to(i)

  ! CHECK: OmpSimpleStandaloneDirective -> llvm::omp::Directive = target update
  ! CHECK-NEXT: OmpClause -> If -> OmpIfClause
  ! CHECK-NEXT: OmpDirectiveNameModifier -> llvm::omp::Directive = target update
  !$omp target update if(target update: cond) to(i)

  ! CHECK: OmpSimpleStandaloneDirective -> llvm::omp::Directive = target enter data
  ! CHECK: OmpClause -> If -> OmpIfClause
  ! CHECK-NEXT: OmpDirectiveNameModifier -> llvm::omp::Directive = target enter data
  !$omp target enter data map(to: i) if(target enter data: cond)

  ! CHECK: OmpSimpleStandaloneDirective -> llvm::omp::Directive = target exit data
  ! CHECK: OmpClause -> If -> OmpIfClause
  ! CHECK-NEXT: OmpDirectiveNameModifier -> llvm::omp::Directive = target exit data
  !$omp target exit data map(from: i) if(target exit data: cond)

  ! CHECK: OmpBlockDirective -> llvm::omp::Directive = target data
  ! CHECK: OmpClause -> If -> OmpIfClause
  ! CHECK-NEXT: OmpDirectiveNameModifier -> llvm::omp::Directive = target data
  !$omp target data map(tofrom: i) if(target data: cond)
  !$omp end target data

  ! CHECK: OmpLoopDirective -> llvm::omp::Directive = target teams distribute parallel do simd
  ! CHECK: OmpClause -> If -> OmpIfClause
  ! CHECK-NEXT: OmpDirectiveNameModifier -> llvm::omp::Directive = target
  ! CHECK: OmpClause -> If -> OmpIfClause
  ! CHECK-NEXT: OmpDirectiveNameModifier -> llvm::omp::Directive = teams
  ! CHECK: OmpClause -> If -> OmpIfClause
  ! CHECK-NEXT: OmpDirectiveNameModifier -> llvm::omp::Directive = parallel
  ! CHECK: OmpClause -> If -> OmpIfClause
  ! CHECK-NEXT: OmpDirectiveNameModifier -> llvm::omp::Directive = simd
  !$omp target teams distribute parallel do simd if(target: cond) &
  !$omp&    if(teams: cond) if(parallel: cond) if(simd: cond)
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do simd

  ! CHECK: OmpBlockDirective -> llvm::omp::Directive = task
  ! CHECK-NEXT: OmpClause -> If -> OmpIfClause
  ! CHECK-NEXT: OmpDirectiveNameModifier -> llvm::omp::Directive = task
  !$omp task if(task: cond)
  !$omp end task

  ! CHECK: OmpLoopDirective -> llvm::omp::Directive = taskloop
  ! CHECK-NEXT: OmpClause -> If -> OmpIfClause
  ! CHECK-NEXT: DirectiveNameModifier -> llvm::omp::Directive = taskloop
  !$omp taskloop if(taskloop: cond)
  do i = 1, 10
  end do
  !$omp end taskloop
end program openmp_parse_if
