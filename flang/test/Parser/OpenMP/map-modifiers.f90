!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=52 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=52 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00(x)
  integer :: x
  !$omp target map(ompx_hold, always, present, close, to: x)
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f00 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TARGET  MAP(OMPX_HOLD, ALWAYS, PRESENT, CLOSE, TO: x)
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | TypeModifier = Ompx_Hold
!PARSE-TREE: | | TypeModifier = Always
!PARSE-TREE: | | TypeModifier = Present
!PARSE-TREE: | | TypeModifier = Close
!PARSE-TREE: | | Type = To
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'

subroutine f01(x)
  integer :: x
  !$omp target map(ompx_hold, always, present, close: x)
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f01 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TARGET  MAP(OMPX_HOLD, ALWAYS, PRESENT, CLOSE: x)
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | TypeModifier = Ompx_Hold
!PARSE-TREE: | | TypeModifier = Always
!PARSE-TREE: | | TypeModifier = Present
!PARSE-TREE: | | TypeModifier = Close
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'

subroutine f02(x)
  integer :: x
  !$omp target map(from: x)
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f02 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TARGET  MAP(FROM: x)
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | Type = From
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'

subroutine f03(x)
  integer :: x
  !$omp target map(x)
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f03 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TARGET  MAP(x)
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'

subroutine f10(x)
  integer :: x
  !$omp target map(ompx_hold always, present, close, to: x)
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f10 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TARGET  MAP(OMPX_HOLD, ALWAYS, PRESENT, CLOSE, TO: x)
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | TypeModifier = Ompx_Hold
!PARSE-TREE: | | TypeModifier = Always
!PARSE-TREE: | | TypeModifier = Present
!PARSE-TREE: | | TypeModifier = Close
!PARSE-TREE: | | Type = To
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'

subroutine f11(x)
  integer :: x
  !$omp target map(ompx_hold, always, present, close: x)
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f11 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TARGET  MAP(OMPX_HOLD, ALWAYS, PRESENT, CLOSE: x)
!UNPARSE:   x=x+1_4
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | TypeModifier = Ompx_Hold
!PARSE-TREE: | | TypeModifier = Always
!PARSE-TREE: | | TypeModifier = Present
!PARSE-TREE: | | TypeModifier = Close
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'

