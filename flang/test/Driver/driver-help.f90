
!--------------------------
! FLANG DRIVER (flang)
!--------------------------
! RUN: %flang -help 2>&1 | FileCheck %s --check-prefix=HELP
! RUN: not %flang -helps 2>&1 | FileCheck %s --check-prefix=ERROR

!----------------------------------------
! FLANG FRONTEND DRIVER (flang -fc1)
!----------------------------------------
! RUN: %flang_fc1 -help 2>&1 | FileCheck %s --check-prefix=HELP-FC1
! RUN: not %flang_fc1 -helps 2>&1 | FileCheck %s --check-prefix=ERROR

!----------------------------------------------------
! EXPECTED OUTPUT FOR FLANG DRIVER (flang)
!----------------------------------------------------
! HELP:USAGE: flang
! HELP-EMPTY:
! HELP-NEXT:OPTIONS:
! HELP-NEXT: -###                   Print (but do not run) the commands to run for this compilation
! HELP-NEXT: -cpp                   Enable predefined and command line preprocessor macros
! HELP-NEXT: -c                     Only run preprocess, compile, and assemble steps
! HELP-NEXT: -D <macro>=<value>     Define <macro> to <value> (or 1 if <value> omitted)
! HELP-NEXT: -E                     Only run the preprocessor
! HELP-NEXT: -falternative-parameter-statement
! HELP-NEXT: Enable the old style PARAMETER statement
! HELP-NEXT: -fbackslash            Specify that backslash in string introduces an escape character
! HELP-NEXT: -fcolor-diagnostics    Enable colors in diagnostics
! HELP-NEXT: -fdefault-double-8     Set the default double precision kind to an 8 byte wide type
! HELP-NEXT: -fdefault-integer-8    Set the default integer kind to an 8 byte wide type
! HELP-NEXT: -fdefault-real-8       Set the default real kind to an 8 byte wide type
! HELP-NEXT: -ffixed-form           Process source files in fixed form
! HELP-NEXT: -ffixed-line-length=<value>
! HELP-NEXT: Use <value> as character line width in fixed mode
! HELP-NEXT: -ffree-form            Process source files in free form
! HELP-NEXT: -fimplicit-none        No implicit typing allowed unless overridden by IMPLICIT statements
! HELP-NEXT: -finput-charset=<value> Specify the default character set for source files
! HELP-NEXT: -fintrinsic-modules-path <dir>
! HELP-NEXT:                        Specify where to find the compiled intrinsic modules
! HELP-NEXT: -flarge-sizes          Use INTEGER(KIND=8) for the result type in size-related intrinsics
! HELP-NEXT: -flogical-abbreviations Enable logical abbreviations
! HELP-NEXT: -fno-automatic         Implies the SAVE attribute for non-automatic local objects in subprograms unless RECURSIVE
! HELP-NEXT: -fno-color-diagnostics  Disable colors in diagnostics
! HELP-NEXT: -fopenacc              Enable OpenACC
! HELP-NEXT: -fopenmp               Parse OpenMP pragmas and generate parallel code.
! HELP-NEXT: -fxor-operator         Enable .XOR. as a synonym of .NEQV.
! HELP-NEXT: -help                  Display available options
! HELP-NEXT: -I <dir>               Add directory to the end of the list of include search paths
! HELP-NEXT: -module-dir <dir>      Put MODULE files in <dir>
! HELP-NEXT: -nocpp                 Disable predefined and command line preprocessor macros
! HELP-NEXT: -o <file>              Write output to <file>
! HELP-NEXT: -pedantic              Warn on language extensions
! HELP-NEXT: -P                     Disable linemarker output in -E mode
! HELP-NEXT: -std=<value>           Language standard to compile for
! HELP-NEXT: -U <macro>             Undefine macro <macro>
! HELP-NEXT: --version              Print version information
! HELP-NEXT: -W<warning>            Enable the specified warning
! HELP-NEXT: -Xflang <arg>          Pass <arg> to the flang compiler

!-------------------------------------------------------------
! EXPECTED OUTPUT FOR FLANG FRONTEND DRIVER (flang -fc1)
!-------------------------------------------------------------
! HELP-FC1:USAGE: flang
! HELP-FC1-EMPTY:
! HELP-FC1-NEXT:OPTIONS:
! HELP-FC1-NEXT: -cpp                   Enable predefined and command line preprocessor macros
! HELP-FC1-NEXT: -D <macro>=<value>     Define <macro> to <value> (or 1 if <value> omitted)
! HELP-FC1-NEXT: -emit-obj Emit native object files
! HELP-FC1-NEXT: -E                     Only run the preprocessor
! HELP-FC1-NEXT: -falternative-parameter-statement
! HELP-FC1-NEXT: Enable the old style PARAMETER statement
! HELP-FC1-NEXT: -fbackslash            Specify that backslash in string introduces an escape character
! HELP-FC1-NEXT: -fdebug-dump-all       Dump symbols and the parse tree after the semantic checks
! HELP-FC1-NEXT: -fdebug-dump-parse-tree-no-sema
! HELP-FC1-NEXT:                        Dump the parse tree (skips the semantic checks)
! HELP-FC1-NEXT: -fdebug-dump-parse-tree Dump the parse tree
! HELP-FC1-NEXT: -fdebug-dump-parsing-log
! HELP-FC1-NEXT:                   Run instrumented parse and dump the parsing log
! HELP-FC1-NEXT: -fdebug-dump-provenance Dump provenance
! HELP-FC1-NEXT: -fdebug-dump-symbols    Dump symbols after the semantic analysis
! HELP-FC1-NEXT: -fdebug-measure-parse-tree
! HELP-FC1-NEXT:                         Measure the parse tree
! HELP-FC1-NEXT: -fdebug-module-writer   Enable debug messages while writing module files
! HELP-FC1-NEXT: -fdebug-pre-fir-tree    Dump the pre-FIR tree
! HELP-FC1-NEXT: -fdebug-unparse-no-sema Unparse and stop (skips the semantic checks)
! HELP-FC1-NEXT: -fdebug-unparse-with-symbols
! HELP-FC1-NEXT:                        Unparse and stop.
! HELP-FC1-NEXT: -fdebug-unparse        Unparse and stop.
! HELP-FC1-NEXT: -fdefault-double-8  Set the default double precision kind to an 8 byte wide type
! HELP-FC1-NEXT: -fdefault-integer-8 Set the default integer kind to an 8 byte wide type
! HELP-FC1-NEXT: -fdefault-real-8    Set the default real kind to an 8 byte wide type
! HELP-FC1-NEXT: -ffixed-form           Process source files in fixed form
! HELP-FC1-NEXT: -ffixed-line-length=<value>
! HELP-FC1-NEXT: Use <value> as character line width in fixed mode
! HELP-FC1-NEXT: -ffree-form            Process source files in free form
! HELP-FC1-NEXT: -fget-definition <value> <value> <value>
! HELP-FC1-NEXT:                        Get the symbol definition from <line> <start-column> <end-column>
! HELP-FC1-NEXT: -fget-symbols-sources   Dump symbols and their source code locations
! HELP-FC1-NEXT: -fimplicit-none        No implicit typing allowed unless overridden by IMPLICIT statements
! HELP-FC1-NEXT: -finput-charset=<value> Specify the default character set for source files
! HELP-FC1-NEXT: -fintrinsic-modules-path <dir>
! HELP-FC1-NEXT:                        Specify where to find the compiled intrinsic modules
! HELP-FC1-NEXT: -flarge-sizes          Use INTEGER(KIND=8) for the result type in size-related intrinsics
! HELP-FC1-NEXT: -flogical-abbreviations Enable logical abbreviations
! HELP-FC1-NEXT: -fno-analyzed-objects-for-unparse
! HELP-FC1-NEXT:                        Do not use the analyzed objects when unparsing
! HELP-FC1-NEXT: -fno-automatic         Implies the SAVE attribute for non-automatic local objects in subprograms unless RECURSIVE
! HELP-FC1-NEXT: -fno-reformat          Dump the cooked character stream in -E mode
! HELP-FC1-NEXT: -fopenacc              Enable OpenACC
! HELP-FC1-NEXT: -fopenmp               Parse OpenMP pragmas and generate parallel code.
! HELP-FC1-NEXT: -fxor-operator         Enable .XOR. as a synonym of .NEQV.
! HELP-FC1-NEXT: -help                  Display available options
! HELP-FC1-NEXT: -init-only             Only execute frontend initialization
! HELP-FC1-NEXT: -I <dir>               Add directory to the end of the list of include search paths
! HELP-FC1-NEXT: -load <dsopath>        Load the named plugin (dynamic shared object)
! HELP-FC1-NEXT: -module-dir <dir>      Put MODULE files in <dir>
! HELP-FC1-NEXT: -module-suffix <suffix> Use <suffix> as the suffix for module files (the default value is `.mod`)
! HELP-FC1-NEXT: -nocpp                 Disable predefined and command line preprocessor macros
! HELP-FC1-NEXT: -o <file>              Write output to <file>
! HELP-FC1-NEXT: -pedantic              Warn on language extensions
! HELP-FC1-NEXT: -plugin <name>         Use the named plugin action instead of the default action (use "help" to list available options)
! HELP-FC1-NEXT: -P                     Disable linemarker output in -E mode
! HELP-FC1-NEXT: -std=<value>           Language standard to compile for
! HELP-FC1-NEXT: -test-io               Run the InputOuputTest action. Use for development and testing only.
! HELP-FC1-NEXT: -U <macro>             Undefine macro <macro>
! HELP-FC1-NEXT: --version              Print version information
! HELP-FC1-NEXT: -W<warning>            Enable the specified warning

!---------------
! EXPECTED ERROR
!---------------
! ERROR: error: unknown argument '-helps'; did you mean '-help'
