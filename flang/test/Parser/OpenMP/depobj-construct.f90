!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=52 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=52 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00
  integer :: x, y
  !$omp depobj(x) depend(in: y)
end

!UNPARSE: SUBROUTINE f00
!UNPARSE:  INTEGER x, y
!UNPARSE: !$OMP DEPOBJ(x) DEPEND(IN: y)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPDepobjConstruct
!PARSE-TREE: | Verbatim
!PARSE-TREE: | OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | OmpClause -> Depend -> OmpDependClause -> TaskDep
!PARSE-TREE: | | Modifier -> OmpTaskDependenceType -> Value = In
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'y'

subroutine f01
  integer :: x
  !$omp depobj(x) update(out)
end

!UNPARSE: SUBROUTINE f01
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP DEPOBJ(x) UPDATE(OUT)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPDepobjConstruct
!PARSE-TREE: | Verbatim
!PARSE-TREE: | OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | OmpClause -> Update -> OmpUpdateClause -> OmpTaskDependenceType -> Value = Out

subroutine f02
  integer :: x
  !$omp depobj(x) destroy(x)
end

!UNPARSE: SUBROUTINE f02
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP DEPOBJ(x) DESTROY(x)
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPDepobjConstruct
!PARSE-TREE: | Verbatim
!PARSE-TREE: | OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | OmpClause -> Destroy -> OmpDestroyClause -> OmpObject -> Designator -> DataRef -> Name = 'x'

subroutine f03
  integer :: x
  !$omp depobj(x) destroy
end

!UNPARSE: SUBROUTINE f03
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP DEPOBJ(x) DESTROY
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPDepobjConstruct
!PARSE-TREE: | Verbatim
!PARSE-TREE: | OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | OmpClause -> Destroy ->
