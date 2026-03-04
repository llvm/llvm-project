!RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s
! CHECK: Program -> ProgramUnit -> MainProgram
! CHECK-NEXT: | ProgramStmt -> Name = 'semicolon'
program semicolon; end
;
subroutine sub; end;
function fn(); end;;
; ;
module mod; end;