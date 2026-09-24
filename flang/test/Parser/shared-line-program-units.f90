!RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s
!RUN: not %flang_fc1 -pedantic -Werror -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s --check-prefix=ERROR
! CHECK: Program -> ProgramUnit -> SubroutineSubprogram
! CHECK: ProgramUnit -> FunctionSubprogram
! CHECK: ProgramUnit -> MainProgram
! ERROR: portability: nonstandard usage: end of program unit not terminated by new line
! ERROR: portability: nonstandard usage: end of program unit not terminated by new line
subroutine sub; end; function fn(); end; program p; end;
! CHECK: ProgramUnit -> SubroutineSubprogram
! CHECK: ProgramUnit -> MainProgram
! CHECK: ProgramUnit -> MainProgram
! CHECK: ProgramUnit -> Module
! ERROR: portability: nonstandard usage: end of program unit not terminated by new line
! ERROR: portability: nonstandard usage: end of program unit not terminated by new line
! ERROR: portability: nonstandard usage: end of program unit not terminated by new line
subroutine sub2; end; end program; end program; module m; end
! CHECK: ProgramUnit -> BlockData
! CHECK: ProgramUnit -> BlockData
! CHECK: ProgramUnit -> BlockData
! ERROR: portability: nonstandard usage: end of program unit not terminated by new line
! ERROR: portability: nonstandard usage: end of program unit not terminated by new line
block data bd; end; block data bd2; end; block data bd3; end
! CHECK: ProgramUnit -> Module
! CHECK: ProgramUnit -> Submodule
! CHECK: ProgramUnit -> Submodule
! ERROR: portability: nonstandard usage: end of program unit not terminated by new line
! ERROR: portability: nonstandard usage: end of program unit not terminated by new line
module sm; end; submodule (sm) sm2; end; submodule (sm:sm2) sm3; end
! CHECK: ProgramUnit -> MainProgram
! CHECK: ProgramUnit -> MainProgram
! ERROR: portability: nonstandard usage: end of program unit not terminated by new line
program p; end; use sm; print *, "Hello, World!"; end
! CHECK: ProgramUnit -> MainProgram
! CHECK: ProgramUnit -> MainProgram
! ERROR: portability: nonstandard usage: end of program unit not terminated by new line
program p; end; use sm; 
    print *, "Hello, World!"; end
! CHECK: ProgramUnit -> MainProgram
! CHECK: ProgramUnit -> MainProgram
! ERROR: portability: nonstandard usage: end of program unit not terminated by new line
program p; end; use sm; print *, "Hello, World!";
end
! CHECK: ProgramUnit -> FunctionSubprogram
! CHECK: ProgramUnit -> MainProgram
function fn(); end
10 print *, "1"; 20 print *, "2";
end program;
! CHECK: ProgramUnit -> FunctionSubprogram
! CHECK: ProgramUnit -> MainProgram
! ERROR: portability: nonstandard usage: end of program unit not terminated by new line
function fn(); end; 10 print *, "1"; 20 print *, "2"; end program;
