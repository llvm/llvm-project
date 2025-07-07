! RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=50 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s 
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=50 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine foo1()
  integer :: x, i
  x = 1
  !$omp parallel do lastprivate(x)
  do i = 1, 100
    x = x + 1
  enddo
end

!UNPARSE: SUBROUTINE foo1
!UNPARSE:  INTEGER x, i
!UNPARSE:   x=1_4
!UNPARSE: !$OMP PARALLEL DO  LASTPRIVATE(x)
!UNPARSE:  DO i=1_4,100_4
!UNPARSE:    x=x+1_4
!UNPARSE:  END DO
!UNPARSE: END SUBROUTINE

!PARSE-TREE: SubroutineStmt
!PARSE-TREE:   Name = 'foo1'
!PARSE-TREE: OmpLoopDirective -> llvm::omp::Directive = parallel do
!PARSE-TREE: OmpClauseList -> OmpClause -> Lastprivate -> OmpLastprivateClause
!PARSE-TREE:   OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: EndSubroutineStmt


subroutine foo2()
  integer :: x, i
  x = 1
  !$omp parallel do lastprivate(conditional: x)
  do i = 1, 100
    x = x + 1
  enddo
end

!UNPARSE: SUBROUTINE foo2
!UNPARSE:  INTEGER x, i
!UNPARSE:   x=1_4
!UNPARSE: !$OMP PARALLEL DO  LASTPRIVATE(CONDITIONAL: x)
!UNPARSE:  DO i=1_4,100_4
!UNPARSE:    x=x+1_4
!UNPARSE:  END DO
!UNPARSE: END SUBROUTINE

!PARSE-TREE: SubroutineStmt
!PARSE-TREE:   Name = 'foo2'
!PARSE-TREE: OmpLoopDirective -> llvm::omp::Directive = parallel do
!PARSE-TREE: OmpClauseList -> OmpClause -> Lastprivate -> OmpLastprivateClause
!PARSE-TREE:   Modifier -> OmpLastprivateModifier -> Value = Conditional
!PARSE-TREE:   OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: EndSubroutineStmt
