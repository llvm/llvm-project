!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=52 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=52 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00
  !$omp target defaultmap(from)
  !$omp end target
end

!UNPARSE: SUBROUTINE f00
!UNPARSE: !$OMP TARGET  DEFAULTMAP(FROM)
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Defaultmap -> OmpDefaultmapClause
!PARSE-TREE: | | ImplicitBehavior = From
!PARSE-TREE: Block

subroutine f01
  !$omp target defaultmap(firstprivate: aggregate)
  !$omp end target
end

!UNPARSE: SUBROUTINE f01
!UNPARSE: !$OMP TARGET  DEFAULTMAP(FIRSTPRIVATE:AGGREGATE)
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Defaultmap -> OmpDefaultmapClause
!PARSE-TREE: | | ImplicitBehavior = Firstprivate
!PARSE-TREE: | | Modifier -> OmpVariableCategory -> Value = Aggregate

subroutine f02
  !$omp target defaultmap(alloc: all)
  !$omp end target
end

!UNPARSE: SUBROUTINE f02
!UNPARSE: !$OMP TARGET  DEFAULTMAP(ALLOC:ALL)
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Defaultmap -> OmpDefaultmapClause
!PARSE-TREE: | | ImplicitBehavior = Alloc
!PARSE-TREE: | | Modifier -> OmpVariableCategory -> Value = All

! Both "all" and "allocatable" are valid, and "all" is a prefix of
! "allocatable". Make sure we parse this correctly.
subroutine f03
  !$omp target defaultmap(alloc: allocatable)
  !$omp end target
end

!UNPARSE: SUBROUTINE f03
!UNPARSE: !$OMP TARGET  DEFAULTMAP(ALLOC:ALLOCATABLE)
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Defaultmap -> OmpDefaultmapClause
!PARSE-TREE: | | ImplicitBehavior = Alloc
!PARSE-TREE: | | Modifier -> OmpVariableCategory -> Value = Allocatable

subroutine f04
  !$omp target defaultmap(tofrom: scalar)
  !$omp end target
end

!UNPARSE: SUBROUTINE f04
!UNPARSE: !$OMP TARGET  DEFAULTMAP(TOFROM:SCALAR)
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: | OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Defaultmap -> OmpDefaultmapClause
!PARSE-TREE: | | ImplicitBehavior = Tofrom
!PARSE-TREE: | | Modifier -> OmpVariableCategory -> Value  = Scalar
