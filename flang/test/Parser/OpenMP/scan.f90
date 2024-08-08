! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

! Check for parsing scan directive
subroutine test_scan_inclusive()
  implicit none
  integer, parameter :: n = 100
  integer a(n), b(n)
  integer x, k

  ! initialization
  x = 0
  do k = 1, n
   a(k) = k
  end do

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
end subroutine
