
!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: %flang --help-hidden 2>&1 | FileCheck %s
! RUN: not %flang  -help-hidden 2>&1 | FileCheck %s --check-prefix=ERROR-FLANG

!----------------------------------------
! FLANG FRONTEND DRIVER (flang-new -fc1)
!----------------------------------------
! RUN: not %flang_fc1 --help-hidden 2>&1 | FileCheck %s --check-prefix=ERROR-FLANG-FC1
! RUN: not %flang_fc1  -help-hidden 2>&1 | FileCheck %s --check-prefix=ERROR-FLANG-FC1

!----------------------------------------------------
! EXPECTED OUTPUT FOR FLANG DRIVER (flang-new)
!----------------------------------------------------
! CHECK:USAGE: flang-new
! CHECK-EMPTY:
! CHECK-NEXT:OPTIONS:
! CHECK-NEXT: -###      Print (but do not run) the commands to run for this compilation
! CHECK-NEXT: -cpp      Enable predefined and command line preprocessor macros
! CHECK-NEXT: -c        Only run preprocess, compile, and assemble steps
! CHECK-NEXT: -D <macro>=<value>     Define <macro> to <value> (or 1 if <value> omitted)
! CHECK-NEXT: -E        Only run the preprocessor
! CHECK-NEXT: -falternative-parameter-statement
! CHECK-NEXT: Enable the old style PARAMETER statement
! CHECK-NEXT: -fbackslash            Specify that backslash in string introduces an escape character
! CHECK-NEXT: -fcolor-diagnostics    Enable colors in diagnostics
! CHECK-NEXT: -fdefault-double-8     Set the default double precision kind to an 8 byte wide type
! CHECK-NEXT: -fdefault-integer-8    Set the default integer kind to an 8 byte wide type
! CHECK-NEXT: -fdefault-real-8       Set the default real kind to an 8 byte wide type
! CHECK-NEXT: -ffixed-form           Process source files in fixed form
! CHECK-NEXT: -ffixed-line-length=<value>
! CHECK-NEXT: Use <value> as character line width in fixed mode
! CHECK-NEXT: -ffree-form            Process source files in free form
! CHECK-NEXT: -fimplicit-none        No implicit typing allowed unless overridden by IMPLICIT statements
! CHECK-NEXT: -finput-charset=<value> Specify the default character set for source files
! CHECK-NEXT: -fintrinsic-modules-path <dir>
! CHECK-NEXT:                        Specify where to find the compiled intrinsic modules
! CHECK-NEXT: -flarge-sizes          Use INTEGER(KIND=8) for the result type in size-related intrinsics
! CHECK-NEXT: -flogical-abbreviations Enable logical abbreviations
! CHECK-NEXT: -fno-color-diagnostics  Disable colors in diagnostics
! CHECK-NEXT: -fopenacc              Enable OpenACC
! CHECK-NEXT: -fopenmp               Parse OpenMP pragmas and generate parallel code.
! CHECK-NEXT: -fxor-operator         Enable .XOR. as a synonym of .NEQV.
! CHECK-NEXT: -help     Display available options
! CHECK-NEXT: -I <dir>               Add directory to the end of the list of include search paths
! CHECK-NEXT: -module-dir <dir>      Put MODULE files in <dir>
! CHECK-NEXT: -nocpp                 Disable predefined and command line preprocessor macros
! CHECK-NEXT: -o <file> Write output to <file>
! CHECK-NEXT: -pedantic              Warn on language extensions
! CHECK-NEXT: -P                     Disable linemarker output in -E mode
! CHECK-NEXT: -std=<value>           Language standard to compile for
! CHECK-NEXT: -U <macro>             Undefine macro <macro>
! CHECK-NEXT: --version Print version information
! CHECK-NEXT: -W<warning>            Enable the specified warning
! CHECK-NEXT: -Xflang <arg>          Pass <arg> to the flang compiler

!-------------------------------------------------------------
! EXPECTED OUTPUT FOR FLANG DRIVER (flang-new)
!-------------------------------------------------------------
! ERROR-FLANG: error: unknown argument '-help-hidden'; did you mean '--help-hidden'?

!-------------------------------------------------------------
! EXPECTED OUTPUT FOR FLANG FRONTEND DRIVER (flang-new -fc1)
!-------------------------------------------------------------
! Frontend driver -help-hidden is not supported
! ERROR-FLANG-FC1: error: unknown argument: '{{.*}}'

