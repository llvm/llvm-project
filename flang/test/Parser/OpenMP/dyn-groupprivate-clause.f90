!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=61 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=61 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00(n)
  implicit none
  integer :: n
  !$omp target dyn_groupprivate(n)
  !$omp end target
end

!UNPARSE: SUBROUTINE f00 (n)
!UNPARSE:  IMPLICIT NONE
!UNPARSE:  INTEGER n
!UNPARSE: !$OMP TARGET DYN_GROUPPRIVATE(n)
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> DynGroupprivate -> OmpDynGroupprivateClause
!PARSE-TREE: | | Scalar -> Integer -> Expr = 'n'
!PARSE-TREE: | | | Designator -> DataRef -> Name = 'n'
!PARSE-TREE: | Flags = None


subroutine f01(n)
  implicit none
  integer :: n
  !$omp target dyn_groupprivate(strict: n)
  !$omp end target
end

!UNPARSE: SUBROUTINE f01 (n)
!UNPARSE:  IMPLICIT NONE
!UNPARSE:  INTEGER n
!UNPARSE: !$OMP TARGET DYN_GROUPPRIVATE(STRICT: n)
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> DynGroupprivate -> OmpDynGroupprivateClause
!PARSE-TREE: | | Modifier -> OmpPrescriptiveness -> Value = Strict
!PARSE-TREE: | | Scalar -> Integer -> Expr = 'n'
!PARSE-TREE: | | | Designator -> DataRef -> Name = 'n'
!PARSE-TREE: | Flags = None


subroutine f02(n)
  implicit none
  integer :: n
  !$omp target dyn_groupprivate(fallback, cgroup: n)
  !$omp end target
end

!UNPARSE: SUBROUTINE f02 (n)
!UNPARSE:  IMPLICIT NONE
!UNPARSE:  INTEGER n
!UNPARSE: !$OMP TARGET DYN_GROUPPRIVATE(FALLBACK, CGROUP: n)
!UNPARSE: !$OMP END TARGET
!UNPARSE: END SUBROUTINE

!PARSE-TREE: OmpBeginDirective
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = target
!PARSE-TREE: | OmpClauseList -> OmpClause -> DynGroupprivate -> OmpDynGroupprivateClause
!PARSE-TREE: | | Modifier -> OmpPrescriptiveness -> Value = Fallback
!PARSE-TREE: | | Modifier -> OmpAccessGroup -> Value = Cgroup
!PARSE-TREE: | | Scalar -> Integer -> Expr = 'n'
!PARSE-TREE: | | | Designator -> DataRef -> Name = 'n'
!PARSE-TREE: | Flags = None
