!RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s
;
! CHECK: Program -> ProgramUnit -> CompilerDirective
!DIR$ unknown_directive empty statement
;
! CHECK: ProgramUnit -> MainProgram
program semicolon; end;
! CHECK: ProgramUnit -> CompilerDirective
    !DIR$ unknown_directive leading space
! CHECK: ProgramUnit -> CompilerDirective
;   !DIR$ unknown_directive leading empty statement
