!RUN: %flang_fc1 -fdebug-unparse-no-sema -fopenmp -fopenmp-version=60 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema -fopenmp -fopenmp-version=60 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00(x)
  integer :: x
  !$omp target map(always, close, delete, present, ompx_hold: x)
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f00 (x)
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP TARGET  MAP(ALWAYS, CLOSE, DELETE, PRESENT, OMPX_HOLD: x)
!UNPARSE:   x = x+1
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | Modifier -> OmpAlwaysModifier -> Value = Always
!PARSE-TREE: | | Modifier -> OmpCloseModifier -> Value = Close
!PARSE-TREE: | | Modifier -> OmpDeleteModifier -> Value = Delete
!PARSE-TREE: | | Modifier -> OmpPresentModifier -> Value = Present
!PARSE-TREE: | | Modifier -> OmpxHoldModifier -> Value = Ompx_Hold
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | bool = 'true'

subroutine f01(x)
  integer :: x
  !$omp target map(self, storage: x)
  x = x + 1
  !$omp end target
end

!UNPARSE: !$OMP TARGET  MAP(SELF, STORAGE: x)
!UNPARSE:   x = x+1
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | Modifier -> OmpSelfModifier -> Value = Self
!PARSE-TREE: | | Modifier -> OmpMapType -> Value = Storage
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | bool = 'true'

subroutine f02(x)
  integer, pointer :: x
  !$omp target map(ref_ptr, to: x)
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f02 (x)
!UNPARSE:  INTEGER, POINTER :: x
!UNPARSE: !$OMP TARGET  MAP(REF_PTR, TO: x)
!UNPARSE:   x = x+1
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | Modifier -> OmpRefModifier -> Value = Ref_Ptr
!PARSE-TREE: | | Modifier -> OmpMapType -> Value = To
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | bool = 'true'

subroutine f03(x)
  integer, pointer :: x
  !$omp target map(ref_ptee, to: x)
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f03 (x)
!UNPARSE:  INTEGER, POINTER :: x
!UNPARSE: !$OMP TARGET  MAP(REF_PTEE, TO: x)
!UNPARSE:   x = x+1
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | Modifier -> OmpRefModifier -> Value = Ref_Ptee
!PARSE-TREE: | | Modifier -> OmpMapType -> Value = To
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | bool = 'true'

subroutine f04(x)
  integer, pointer :: x
  !$omp target map(ref_ptr_ptee, to: x)
  x = x + 1
  !$omp end target
end

!UNPARSE: SUBROUTINE f04 (x)
!UNPARSE:  INTEGER, POINTER :: x
!UNPARSE: !$OMP TARGET  MAP(REF_PTR_PTEE, TO: x)
!UNPARSE:   x = x+1
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Map -> OmpMapClause
!PARSE-TREE: | | Modifier -> OmpRefModifier -> Value = Ref_Ptr_Ptee
!PARSE-TREE: | | Modifier -> OmpMapType -> Value = To
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | bool = 'true'
