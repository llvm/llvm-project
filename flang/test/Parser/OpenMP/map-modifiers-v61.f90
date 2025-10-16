!RUN: %flang_fc1 -fdebug-unparse-no-sema -fopenmp -fopenmp-version=61 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema -fopenmp -fopenmp-version=61 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00(x)
  integer, pointer :: x
  !$omp target map(attach(always): x)
  !$omp end target
end

!UNPARSE: SUBROUTINE f00 (x)
!UNPARSE:  INTEGER, POINTER :: x
!UNPARSE: !$OMP TARGET MAP(ATTACH(ALWAYS): x)
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | Modifier -> OmpAttachModifier -> Value = Always
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | bool = 'true'
!PARSE-TREE: | Flags = None


subroutine f01(x)
  integer, pointer :: x
  !$omp target map(attach(auto): x)
  !$omp end target
end

!UNPARSE: SUBROUTINE f01 (x)
!UNPARSE:  INTEGER, POINTER :: x
!UNPARSE: !$OMP TARGET MAP(ATTACH(AUTO): x)
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | Modifier -> OmpAttachModifier -> Value = Auto
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | bool = 'true'
!PARSE-TREE: | Flags = None


subroutine f02(x)
  integer, pointer :: x
  !$omp target map(attach(never): x)
  !$omp end target
end

!UNPARSE: SUBROUTINE f02 (x)
!UNPARSE:  INTEGER, POINTER :: x
!UNPARSE: !$OMP TARGET MAP(ATTACH(NEVER): x)
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | Modifier -> OmpAttachModifier -> Value = Never
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | bool = 'true'
!PARSE-TREE: | Flags = None
