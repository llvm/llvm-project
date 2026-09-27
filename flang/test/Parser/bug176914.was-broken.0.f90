!RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s
! CHECK: Program -> ProgramUnit -> MainProgram
program semicolon; end
;
