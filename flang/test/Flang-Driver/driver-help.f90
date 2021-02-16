! REQUIRES: new-flang-driver

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: %flang-new -help 2>&1 | FileCheck %s --check-prefix=HELP
! RUN: not %flang-new -helps 2>&1 | FileCheck %s --check-prefix=ERROR

!----------------------------------------
! FLANG FRONTEND DRIVER (flang-new -fc1)
!----------------------------------------
! RUN: %flang-new -fc1 -help 2>&1 | FileCheck %s --check-prefix=HELP-FC1
! RUN: not %flang-new -fc1 -helps 2>&1 | FileCheck %s --check-prefix=ERROR

!----------------------------------------------------
! EXPECTED OUTPUT FOR FLANG DRIVER (flang-new)
!----------------------------------------------------
! HELP:USAGE: flang-new
! HELP-EMPTY:
! HELP-NEXT:OPTIONS:
! HELP-NEXT: -###                   Print (but do not run) the commands to run for this compilation
! HELP-NEXT: -c                     Only run preprocess, compile, and assemble steps
! HELP-NEXT: -D <macro>=<value>     Define <macro> to <value> (or 1 if <value> omitted)
! HELP-NEXT: -E                     Only run the preprocessor
! HELP-NEXT: -fcolor-diagnostics    Enable colors in diagnostics
! HELP-NEXT: -ffixed-form           Process source files in fixed form
! HELP-NEXT: -ffixed-line-length=<value>
! HELP-NEXT: Use <value> as character line width in fixed mode
! HELP-NEXT: -ffree-form            Process source files in free form
! HELP-NEXT: -fno-color-diagnostics Disable colors in diagnostics
! HELP-NEXT: -fopenacc              Enable OpenACC
! HELP-NEXT: -fopenmp               Parse OpenMP pragmas and generate parallel code.
! HELP-NEXT: -help                  Display available options
! HELP-NEXT: -I <dir>               Add directory to the end of the list of include search paths
! HELP-NEXT: -module-dir <dir>      Put MODULE files in <dir>
! HELP-NEXT: -o <file>              Write output to <file>
! HELP-NEXT: -U <macro>             Undefine macro <macro>
! HELP-NEXT: --version              Print version information

!-------------------------------------------------------------
! EXPECTED OUTPUT FOR FLANG FRONTEND DRIVER (flang-new -fc1)
!-------------------------------------------------------------
! HELP-FC1:USAGE: flang-new
! HELP-FC1-EMPTY:
! HELP-FC1-NEXT:OPTIONS:
! HELP-FC1-NEXT: -D <macro>=<value>     Define <macro> to <value> (or 1 if <value> omitted)
! HELP-FC1-NEXT: -emit-obj Emit native object files
! HELP-FC1-NEXT: -E                     Only run the preprocessor
! HELP-FC1-NEXT: -fdebug-unparse-with-symbols
! HELP-FC1-NEXT:                        Unparse and stop.
! HELP-FC1-NEXT: -fdebug-unparse        Unparse and stop.
! HELP-FC1-NEXT: -ffixed-form           Process source files in fixed form
! HELP-FC1-NEXT: -ffixed-line-length=<value>
! HELP-FC1-NEXT: Use <value> as character line width in fixed mode
! HELP-FC1-NEXT: -ffree-form            Process source files in free form
! HELP-FC1-NEXT: -fopenacc              Enable OpenACC
! HELP-FC1-NEXT: -fopenmp               Parse OpenMP pragmas and generate parallel code.
! HELP-FC1-NEXT: -help                  Display available options
! HELP-FC1-NEXT: -I <dir>               Add directory to the end of the list of include search paths
! HELP-FC1-NEXT: -module-dir <dir>      Put MODULE files in <dir>
! HELP-FC1-NEXT: -o <file>              Write output to <file>
! HELP-FC1-NEXT: -U <macro>             Undefine macro <macro>
! HELP-FC1-NEXT: --version              Print version information

!---------------
! EXPECTED ERROR
!---------------
! ERROR: error: unknown argument '-helps'; did you mean '-help'
