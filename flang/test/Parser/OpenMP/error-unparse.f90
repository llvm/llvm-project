! RUN: %flang_fc1  -fopenmp-version=51 -fopenmp -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s
! RUN: %flang_fc1  -fopenmp-version=51 -fopenmp -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s --check-prefix="PARSE-TREE"
program main
  character(*), parameter :: message = "This is an error"
  !CHECK: !$OMP ERROR AT(COMPILATION) SEVERITY(WARNING) MESSAGE("some message here")
  !PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPErrorConstruct
  !PARSE-TREE: OmpClauseList -> OmpClause -> At -> OmpAtClause -> ActionTime = Compilation
  !PARSE-TREE: OmpClause -> Severity -> OmpSeverityClause -> Severity = Warning
  !PARSE-TREE:  OmpClause -> Message -> OmpMessageClause -> Expr -> LiteralConstant -> CharLiteralConstant
  !$omp error at(compilation) severity(warning) message("some message here")
  !CHECK: !$OMP ERROR AT(COMPILATION) SEVERITY(FATAL) MESSAGE(message)
  !PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPErrorConstruct
  !PARSE-TREE: OmpClauseList -> OmpClause -> At -> OmpAtClause -> ActionTime = Compilation
  !PARSE-TREE: OmpClause -> Severity -> OmpSeverityClause -> Severity = Fatal
  !PARSE-TREE:  OmpClause -> Message -> OmpMessageClause -> Expr -> Designator -> DataRef -> Name = 'message'
  !$omp error at(compilation) severity(fatal) message(message)
  !CHECK: !$OMP ERROR AT(EXECUTION) SEVERITY(FATAL) MESSAGE(message)
  !PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPErrorConstruct
  !PARSE-TREE: OmpClauseList -> OmpClause -> At -> OmpAtClause -> ActionTime = Execution
  !PARSE-TREE: OmpClause -> Severity -> OmpSeverityClause -> Severity = Fatal
  !PARSE-TREE:  OmpClause -> Message -> OmpMessageClause -> Expr -> Designator ->  DataRef -> Name = 'message'
  !$omp error at(EXECUTION) severity(fatal) message(message)
end program main
