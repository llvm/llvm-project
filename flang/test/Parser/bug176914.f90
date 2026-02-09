!RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s
! CHECK: Program -> ProgramUnit -> MainProgram
! CHECK-NEXT: | ProgramStmt -> Name = 'semicolon'
; program semicolon
use semi;
IMPLICIT NONE  ;  
testvar2 = testvar
;; ; intvar = testvar + testvar2;intvar2=intvar;
print *,testvar , testvar2, intvar, intvar2
! CHECK: | EndProgramStmt
! CHECK: ProgramUnit -> SubroutineSubprogram
! CHECK-NEXT: | SubroutineStmt
! CHECK-NEXT: | | Name = 'test'
end ; ; subroutine test( arg1, arg2 );real arg1, arg2;;;; arg1=arg2; ; ; end;;;
;;
! CHECK: | EndSubroutineStmt