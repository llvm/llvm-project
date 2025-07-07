!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=52 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=52 %s | FileCheck --check-prefix="PARSE-TREE" %s

module m
  integer :: x, y

  !$omp declare target to(x, y)
end

!UNPARSE: MODULE m
!UNPARSE:  INTEGER x, y
!UNPARSE: !$OMP DECLARE TARGET  TO(x,y)
!UNPARSE: END MODULE

!PARSE-TREE: OmpDeclareTargetSpecifier -> OmpDeclareTargetWithClause -> OmpClauseList -> OmpClause -> To -> OmpToClause
!PARSE-TREE: | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | OmpObject -> Designator -> DataRef -> Name = 'y'
!PARSE-TREE: | bool = 'true'

