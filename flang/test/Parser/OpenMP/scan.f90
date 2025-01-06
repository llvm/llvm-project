! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

! Check for parsing scan directive
subroutine test_scan(n, a, b)
  implicit none
  integer n
  integer a(n), b(n)
  integer x,y,k

  ! a(k) is included in the computation of producing results in b(k)
  !$omp parallel do simd reduction(inscan,+: x)
  do k = 1, n
    x = x + a(k)
    !PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct
    !PARSE-TREE-NEXT: OmpSimpleStandaloneDirective -> llvm::omp::Directive = scan
    !PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Inclusive -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
    !CHECK: !$omp scan inclusive(x)
    !$omp scan inclusive(x)
      b(k) = x
  end do

  ! a(k) is not included in the computation of producing results in b(k)
  !$omp parallel do simd reduction(inscan,+: x)
  do k = 1, n
    b(k) = x
    !PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct
    !PARSE-TREE-NEXT: OmpSimpleStandaloneDirective -> llvm::omp::Directive = scan
    !PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Exclusive -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
    !CHECK: !$omp scan exclusive(x)
    !$omp scan exclusive(x)
    x = x + a(k)
  end do

  !$omp parallel do simd reduction(inscan,+: x, y)
  do k = 1, n
    x = x + a(k)
    !PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct
    !PARSE-TREE-NEXT: OmpSimpleStandaloneDirective -> llvm::omp::Directive = scan
    !PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Inclusive -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
    !PARSE-TREE-NEXT: OmpObject -> Designator -> DataRef -> Name = 'y'
    !CHECK: !$omp scan inclusive(x,y)
    !$omp scan inclusive(x, y)
      b(k) = x
  end do

  !$omp parallel do simd reduction(inscan,+: x, y)
  do k = 1, n
    x = x + a(k)
    !PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct
    !PARSE-TREE-NEXT: OmpSimpleStandaloneDirective -> llvm::omp::Directive = scan
    !PARSE-TREE-NEXT: OmpClauseList -> OmpClause -> Exclusive -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
    !PARSE-TREE-NEXT: OmpObject -> Designator -> DataRef -> Name = 'y'
    !CHECK: !$omp scan exclusive(x,y)
    !$omp scan exclusive(x, y)
      b(k) = x
  end do
end subroutine
