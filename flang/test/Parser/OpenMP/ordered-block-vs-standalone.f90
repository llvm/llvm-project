! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=45 %s | FileCheck %s

! Check that standalone ORDERED is successfully distinguished form block associated ORDERED

! CHECK:       | SubroutineStmt
! CHECK-NEXT:  | | Name = 'standalone'
subroutine standalone
  integer :: x(10, 10)
  do i = 1, 10
    do j = 1,10
      ! CHECK:      OpenMPConstruct -> OpenMPStandaloneConstruct
      ! CHECK-NEXT: | OmpDirectiveName -> llvm::omp::Directive = ordered
      ! CHECK-NEXT: | OmpClauseList ->
      ! CHECK-NEXT: | Flags = None
      !$omp ordered
      x(i, j) = i + j
    end do
  end do
endsubroutine

! CHECK:       | SubroutineStmt
! CHECK-NEXT:  | | Name = 'strict_block'
subroutine strict_block
  integer :: x(10, 10)
  integer :: tmp
  do i = 1, 10
    do j = 1,10
      ! CHECK:      OpenMPConstruct -> OmpBlockConstruct
      ! CHECK-NEXT: | OmpBeginDirective
      ! CHECK-NEXT: | | OmpDirectiveName -> llvm::omp::Directive = ordered
      ! CHECK-NEXT: | | OmpClauseList ->
      ! CHECK-NEXT: | | Flags = None
      !$omp ordered
      block
        tmp = i + j
        x(i, j) = tmp
      end block
    end do
  end do
endsubroutine

! CHECK:       | SubroutineStmt
! CHECK-NEXT:  | | Name = 'loose_block'
subroutine loose_block
  integer :: x(10, 10)
  integer :: tmp
  do i = 1, 10
    do j = 1,10
      ! CHECK:      OpenMPConstruct -> OmpBlockConstruct
      ! CHECK-NEXT: | OmpBeginDirective
      ! CHECK-NEXT: | | OmpDirectiveName -> llvm::omp::Directive = ordered
      ! CHECK-NEXT: | | OmpClauseList ->
      ! CHECK-NEXT: | | Flags = None
      !$omp ordered
        tmp = i + j
        x(i, j) = tmp
      !$omp end ordered
    end do
  end do
endsubroutine
