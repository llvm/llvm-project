! RUN: %flang_fc1  -fopenmp-version=51 -fopenmp -fdebug-unparse %s 2>&1 | FileCheck %s
! RUN: %flang_fc1  -fopenmp-version=51 -fopenmp -fdebug-dump-parse-tree %s 2>&1 | FileCheck %s --check-prefix="PARSE-TREE"

! The AT(COMPILATION) form emits its diagnostic during semantics, so SEVERITY(FATAL)
! would halt compilation before any unparse/parse-tree output is produced. Only
! SEVERITY(WARNING) is used with AT(COMPILATION) here; SEVERITY(FATAL) is exercised
! with AT(EXECUTION), which is not diagnosed until lowering. The compilation-time
! diagnostics themselves are covered by Semantics/OpenMP/error.f90.
program main
  character(*), parameter :: message = "This is an error"
  !CHECK: !$OMP ERROR AT(COMPILATION) SEVERITY(WARNING) MESSAGE("some message here")
  !PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpUtilityDirective -> OmpErrorDirective
  !PARSE-TREE: OmpClauseList -> OmpClause -> At -> OmpAtClause -> ActionTime = Compilation
  !PARSE-TREE: OmpClause -> Severity -> OmpSeverityClause -> SevLevel = Warning
  !PARSE-TREE:  OmpClause -> Message -> OmpMessageClause -> Expr = '"some message here"'
  !PARSE-TREE:  LiteralConstant -> CharLiteralConstant
  !PARSE-TREE:  string = 'some message here'
  !$omp error at(compilation) severity(warning) message("some message here")
  !CHECK: !$OMP ERROR AT(EXECUTION) SEVERITY(FATAL) MESSAGE("This is an error")
  !PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OmpUtilityDirective -> OmpErrorDirective
  !PARSE-TREE: OmpClauseList -> OmpClause -> At -> OmpAtClause -> ActionTime = Execution
  !PARSE-TREE: OmpClause -> Severity -> OmpSeverityClause -> SevLevel = Fatal
  !PARSE-TREE:  OmpClause -> Message -> OmpMessageClause -> Expr = '"This is an error"'
  !PARSE-TREE:  Designator ->  DataRef -> Name = 'message'
  !$omp error at(EXECUTION) severity(fatal) message(message)
end program main
