! REQUIRES: new-flang-driver

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: %flang-new --help-hidden 2>&1 | FileCheck %s
! RUN: not %flang-new  -help-hidden 2>&1 | FileCheck %s --check-prefix=ERROR-FLANG

!----------------------------------------
! FLANG FRONTEND DRIVER (flang-new -fc1)
!----------------------------------------
! RUN: not %flang-new -fc1 --help-hidden 2>&1 | FileCheck %s --check-prefix=ERROR-FLANG-FC1
! RUN: not %flang-new -fc1  -help-hidden 2>&1 | FileCheck %s --check-prefix=ERROR-FLANG-FC1

!----------------------------------------------------
! EXPECTED OUTPUT FOR FLANG DRIVER (flang-new)
!----------------------------------------------------
! CHECK:USAGE: flang-new
! CHECK-EMPTY:
! CHECK-NEXT:OPTIONS:
! CHECK-NEXT: -###      Print (but do not run) the commands to run for this compilation
! CHECK-NEXT: -c        Only run preprocess, compile, and assemble steps
! CHECK-NEXT: -D <macro>=<value>     Define <macro> to <value> (or 1 if <value> omitted)
! CHECK-NEXT: -E        Only run the preprocessor
! CHECK-NEXT: -fcolor-diagnostics    Enable colors in diagnostics
! CHECK-NEXT: -ffixed-form           Process source files in fixed form
! CHECK-NEXT: -ffixed-line-length=<value>
! CHECK-NEXT: Use <value> as character line width in fixed mode
! CHECK-NEXT: -ffree-form            Process source files in free form
! CHECK-NEXT: -fno-color-diagnostics Disable colors in diagnostics
! CHECK-NEXT: -fopenacc              Enable OpenACC
! CHECK-NEXT: -fopenmp               Parse OpenMP pragmas and generate parallel code.
! CHECK-NEXT: -help     Display available options
! CHECK-NEXT: -I <dir>               Add directory to the end of the list of include search paths
! CHECK-NEXT: -module-dir <dir>      Put MODULE files in <dir>
! CHECK-NEXT: -o <file> Write output to <file>
! CHECK-NEXT: -test-io  Run the InputOuputTest action. Use for development and testing only.
! CHECK-NEXT: -U <macro>             Undefine macro <macro>
! CHECK-NEXT: --version Print version information

!-------------------------------------------------------------
! EXPECTED OUTPUT FOR FLANG DRIVER (flang-new)
!-------------------------------------------------------------
! ERROR-FLANG: error: unknown argument '-help-hidden'; did you mean '--help-hidden'?

!-------------------------------------------------------------
! EXPECTED OUTPUT FOR FLANG FRONTEND DRIVER (flang-new -fc1)
!-------------------------------------------------------------
! Frontend driver -help-hidden is not supported
! ERROR-FLANG-FC1: error: unknown argument: '{{.*}}'

