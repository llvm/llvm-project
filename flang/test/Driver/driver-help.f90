
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

! HELP:USAGE: flang
! HELP-EMPTY:
! HELP-NEXT:OPTIONS:
! HELP-NEXT: -###                    Print (but do not run) the commands to run for this compilation
! HELP-NEXT: -cpp                    Enable predefined and command line preprocessor macros
! HELP-NEXT: -c                      Only run preprocess, compile, and assemble steps
! HELP-NEXT: -dumpmachine            Display the compiler's target processor
! HELP-NEXT: -dumpversion            Display the version of the compiler
! HELP-NEXT: -D <macro>=<value>      Define <macro> to <value> (or 1 if <value> omitted)
! HELP-NEXT: -emit-llvm              Use the LLVM representation for assembler and object files
! HELP-NEXT: -E                      Only run the preprocessor
! HELP-NEXT: -falternative-parameter-statement
! HELP-NEXT:                         Enable the old style PARAMETER statement
! HELP-NEXT: -fapprox-func           Allow certain math function calls to be replaced with an approximately equivalent calculation
! HELP-NEXT: -fbackslash             Specify that backslash in string introduces an escape character
! HELP-NEXT: -fcolor-diagnostics     Enable colors in diagnostics
! HELP-NEXT: -fconvert=<value>       Set endian conversion of data for unformatted files
! HELP-NEXT: -fdefault-double-8      Set the default double precision kind to an 8 byte wide type
! HELP-NEXT: -fdefault-integer-8     Set the default integer and logical kind to an 8 byte wide type
! HELP-NEXT: -fdefault-real-8        Set the default real kind to an 8 byte wide type
! HELP-NEXT: -ffast-math             Allow aggressive, lossy floating-point optimizations
! HELP-NEXT: -ffixed-form            Process source files in fixed form
! HELP-NEXT: -ffixed-line-length=<value>
! HELP-NEXT:                         Use <value> as character line width in fixed mode
! HELP-NEXT: -ffp-contract=<value>   Form fused FP ops (e.g. FMAs)
! HELP-NEXT: -ffree-form             Process source files in free form
! HELP-NEXT: -fhonor-infinities      Specify that floating-point optimizations are not allowed that assume arguments and results are not +-inf.
! HELP-NEXT: -fhonor-nans            Specify that floating-point optimizations are not allowed that assume arguments and results are not NANs.
! HELP-NEXT: -fimplicit-none         No implicit typing allowed unless overridden by IMPLICIT statements
! HELP-NEXT: -finput-charset=<value> Specify the default character set for source files
! HELP-NEXT: -fintegrated-as         Enable the integrated assembler
! HELP-NEXT: -fintrinsic-modules-path <dir>
! HELP-NEXT:                         Specify where to find the compiled intrinsic modules
! HELP-NEXT: -flarge-sizes           Use INTEGER(KIND=8) for the result type in size-related intrinsics
! HELP-NEXT: -flogical-abbreviations Enable logical abbreviations
! HELP-NEXT: -flto=auto              Enable LTO in 'full' mode
! HELP-NEXT: -flto=jobserver         Enable LTO in 'full' mode
! HELP-NEXT: -flto=<value>           Set LTO mode
! HELP-NEXT: -flto                   Enable LTO in 'full' mode
! HELP-NEXT: -fno-automatic          Implies the SAVE attribute for non-automatic local objects in subprograms unless RECURSIVE
! HELP-NEXT: -fno-color-diagnostics  Disable colors in diagnostics
! HELP-NEXT: -fno-integrated-as      Disable the integrated assembler
! HELP-NEXT: -fno-lto                Disable LTO mode (default)
! HELP-NEXT: -fno-ppc-native-vector-element-order
! HELP-NEXT:                         Specifies PowerPC non-native vector element order
! HELP-NEXT: -fno-signed-zeros       Allow optimizations that ignore the sign of floating point zeros
! HELP-NEXT: -fno-stack-arrays       Allocate array temporaries on the heap (default)
! HELP-NEXT: -fno-version-loops-for-stride
! HELP-NEXT:                         Do not create unit-strided loops (default)
! HELP-NEXT: -fopenacc               Enable OpenACC
! HELP-NEXT: -fopenmp-target-debug   Enable debugging in the OpenMP offloading device RTL
! HELP-NEXT: -fopenmp-targets=<value>
! HELP-NEXT:                         Specify comma-separated list of triples OpenMP offloading targets to be supported
! HELP-NEXT: -fopenmp-version=<value>
! HELP-NEXT:                         Set OpenMP version (e.g. 45 for OpenMP 4.5, 51 for OpenMP 5.1). Default value is 51 for Clang
! HELP-NEXT: -fopenmp                Parse OpenMP pragmas and generate parallel code.
! HELP-NEXT: -foptimization-record-file=<file>
! HELP-NEXT:                         Specify the output name of the file containing the optimization remarks. Implies -fsave-optimization-record. On Darwin platforms, this cannot be used with multiple -arch <arch> options.
! HELP-NEXT: -foptimization-record-passes=<regex>
! HELP-NEXT:                         Only include passes which match a specified regular expression in the generated optimization record (by default, include all passes)
! HELP-NEXT: -fpass-plugin=<dsopath> Load pass plugin from a dynamic shared object file (only with new pass manager).
! HELP-NEXT: -fppc-native-vector-element-order
! HELP-NEXT:                         Specifies PowerPC native vector element order (default)
! HELP-NEXT: -freciprocal-math       Allow division operations to be reassociated
! HELP-NEXT: -fropi                  Generate read-only position independent code (ARM only)
! HELP-NEXT: -frwpi                  Generate read-write position independent code (ARM only)
! HELP-NEXT: -fsave-optimization-record=<format>
! HELP-NEXT:                         Generate an optimization record file in a specific format
! HELP-NEXT: -fsave-optimization-record
! HELP-NEXT:                         Generate a YAML optimization record file
! HELP-NEXT: -fstack-arrays          Attempt to allocate array temporaries on the stack, no matter their size
! HELP-NEXT: -fsyntax-only           Run the preprocessor, parser and semantic analysis stages
! HELP-NEXT: -funderscoring          Appends one trailing underscore to external names
! HELP-NEXT: -fversion-loops-for-stride
! HELP-NEXT:                         Create unit-strided versions of loops
! HELP-NEXT: -fxor-operator          Enable .XOR. as a synonym of .NEQV.
! HELP-NEXT: -gline-directives-only  Emit debug line info directives only
! HELP-NEXT: -gline-tables-only      Emit debug line number tables only
! HELP-NEXT: -g                      Generate source-level debug information
! HELP-NEXT: --help-hidden           Display help for hidden options
! HELP-NEXT: -help                   Display available options
! HELP-NEXT: -I <dir>                Add directory to the end of the list of include search paths
! HELP-NEXT: -L <dir>                Add directory to library search path
! HELP-NEXT: -march=<value>          For a list of available architectures for the target use '-mcpu=help'
! HELP-NEXT: -mcpu=<value>           For a list of available CPUs for the target use '-mcpu=help'
! HELP-NEXT: -mllvm=<arg>            Alias for -mllvm
! HELP-NEXT: -mllvm <value>          Additional arguments to forward to LLVM's option processing
! HELP-NEXT: -mmlir <value>          Additional arguments to forward to MLIR's option processing
! HELP-NEXT: -module-dir <dir>       Put MODULE files in <dir>
! HELP-NEXT: -msve-vector-bits=<value>
! HELP-NEXT:                          Specify the size in bits of an SVE vector register. Defaults to the vector length agnostic value of "scalable". (AArch64 only)
! HELP-NEXT: --no-offload-arch=<value>
! HELP-NEXT:                         Remove CUDA/HIP offloading device architecture (e.g. sm_35, gfx906) from the list of devices to compile for. 'all' resets the list to its default value.
! HELP-NEXT: -nocpp                  Disable predefined and command line preprocessor macros
! HELP-NEXT: --offload-arch=<value>  Specify an offloading device architecture for CUDA, HIP, or OpenMP. (e.g. sm_35). If 'native' is used the compiler will detect locally installed architectures. For HIP offloading, the device architecture can be followed by target ID features delimited by a colon (e.g. gfx908:xnack+:sramecc-). May be specified more than once.
! HELP-NEXT: --offload-device-only   Only compile for the offloading device.
! HELP-NEXT: --offload-host-device   Compile for both the offloading host and device (default).
! HELP-NEXT: --offload-host-only     Only compile for the offloading host.
! HELP-NEXT: -o <file>               Write output to <file>
! HELP-NEXT: -pedantic               Warn on language extensions
! HELP-NEXT: -print-effective-triple Print the effective target triple
! HELP-NEXT: -print-target-triple    Print the normalized target triple
! HELP-NEXT: -P                      Disable linemarker output in -E mode
! HELP-NEXT: -Rpass-analysis=<value> Report transformation analysis from optimization passes whose name matches the given POSIX regular expression
! HELP-NEXT: -Rpass-missed=<value>   Report missed transformations by optimization passes whose name matches the given POSIX regular expression
! HELP-NEXT: -Rpass=<value>          Report transformations performed by optimization passes whose name matches the given POSIX regular expression
! HELP-NEXT: -R<remark>              Enable the specified remark
! HELP-NEXT: -save-temps=<value>     Save intermediate compilation results.
! HELP-NEXT: -save-temps             Save intermediate compilation results
! HELP-NEXT: -std=<value>            Language standard to compile for
! HELP-NEXT: -S                      Only run preprocess and compilation steps
! HELP-NEXT: --target=<value>        Generate code for the given target
! HELP-NEXT: -U <macro>              Undefine macro <macro>
! HELP-NEXT: --version               Print version information
! HELP-NEXT: -v                      Show commands to run and use verbose output
! HELP-NEXT: -Wl,<arg>               Pass the comma separated arguments in <arg> to the linker
! HELP-NEXT: -W<warning>             Enable the specified warning
! HELP-NEXT: -Xflang <arg>           Pass <arg> to the flang compiler
! HELP-NEXT: -x <language>           Treat subsequent input files as having type <language>


! HELP-FC1:USAGE: flang
! HELP-FC1-EMPTY:
! HELP-FC1-NEXT:OPTIONS:
! HELP-FC1-NEXT: -cpp                    Enable predefined and command line preprocessor macros
! HELP-FC1-NEXT: -D <macro>=<value>      Define <macro> to <value> (or 1 if <value> omitted)
! HELP-FC1-NEXT: -emit-fir               Build the parse tree, then lower it to FIR
! HELP-FC1-NEXT: -emit-hlfir             Build the parse tree, then lower it to HLFIR
! HELP-FC1-NEXT: -emit-llvm-bc           Build ASTs then convert to LLVM, emit .bc file
! HELP-FC1-NEXT: -emit-llvm              Use the LLVM representation for assembler and object files
! HELP-FC1-NEXT: -emit-obj               Emit native object files
! HELP-FC1-NEXT: -E                      Only run the preprocessor
! HELP-FC1-NEXT: -falternative-parameter-statement
! HELP-FC1-NEXT:                         Enable the old style PARAMETER statement
! HELP-FC1-NEXT: -fapprox-func           Allow certain math function calls to be replaced with an approximately equivalent calculation
! HELP-FC1-NEXT: -fbackslash             Specify that backslash in string introduces an escape character
! HELP-FC1-NEXT: -fcolor-diagnostics     Enable colors in diagnostics
! HELP-FC1-NEXT: -fconvert=<value>       Set endian conversion of data for unformatted files
! HELP-FC1-NEXT: -fdebug-dump-all        Dump symbols and the parse tree after the semantic checks
! HELP-FC1-NEXT: -fdebug-dump-parse-tree-no-sema
! HELP-FC1-NEXT:                         Dump the parse tree (skips the semantic checks)
! HELP-FC1-NEXT: -fdebug-dump-parse-tree Dump the parse tree
! HELP-FC1-NEXT: -fdebug-dump-parsing-log
! HELP-FC1-NEXT:                         Run instrumented parse and dump the parsing log
! HELP-FC1-NEXT: -fdebug-dump-pft        Dump the pre-fir parse tree
! HELP-FC1-NEXT: -fdebug-dump-provenance Dump provenance
! HELP-FC1-NEXT: -fdebug-dump-symbols    Dump symbols after the semantic analysis
! HELP-FC1-NEXT: -fdebug-measure-parse-tree
! HELP-FC1-NEXT:                         Measure the parse tree
! HELP-FC1-NEXT: -fdebug-module-writer   Enable debug messages while writing module files
! HELP-FC1-NEXT: -fdebug-pass-manager    Prints debug information for the new pass manager
! HELP-FC1-NEXT: -fdebug-pre-fir-tree    Dump the pre-FIR tree
! HELP-FC1-NEXT: -fdebug-unparse-no-sema Unparse and stop (skips the semantic checks)
! HELP-FC1-NEXT: -fdebug-unparse-with-symbols
! HELP-FC1-NEXT:                         Unparse and stop.
! HELP-FC1-NEXT: -fdebug-unparse         Unparse and stop.
! HELP-FC1-NEXT: -fdefault-double-8      Set the default double precision kind to an 8 byte wide type
! HELP-FC1-NEXT: -fdefault-integer-8     Set the default integer and logical kind to an 8 byte wide type
! HELP-FC1-NEXT: -fdefault-real-8        Set the default real kind to an 8 byte wide type
! HELP-FC1-NEXT: -fembed-offload-object=<value>
! HELP-FC1-NEXT:                         Embed Offloading device-side binary into host object file as a section.
! HELP-FC1-NEXT: -ffast-math             Allow aggressive, lossy floating-point optimizations
! HELP-FC1-NEXT: -ffixed-form            Process source files in fixed form
! HELP-FC1-NEXT: -ffixed-line-length=<value>
! HELP-FC1-NEXT:                         Use <value> as character line width in fixed mode
! HELP-FC1-NEXT: -ffp-contract=<value>   Form fused FP ops (e.g. FMAs)
! HELP-FC1-NEXT: -ffree-form             Process source files in free form
! HELP-FC1-NEXT: -fget-definition <value> <value> <value>
! HELP-FC1-NEXT:                         Get the symbol definition from <line> <start-column> <end-column>
! HELP-FC1-NEXT: -fget-symbols-sources   Dump symbols and their source code locations
! HELP-FC1-NEXT: -fimplicit-none         No implicit typing allowed unless overridden by IMPLICIT statements
! HELP-FC1-NEXT: -finput-charset=<value> Specify the default character set for source files
! HELP-FC1-NEXT: -fintrinsic-modules-path <dir>
! HELP-FC1-NEXT:                         Specify where to find the compiled intrinsic modules
! HELP-FC1-NEXT: -flarge-sizes           Use INTEGER(KIND=8) for the result type in size-related intrinsics
! HELP-FC1-NEXT: -flogical-abbreviations Enable logical abbreviations
! HELP-FC1-NEXT: -flto=<value>           Set LTO mode
! HELP-FC1-NEXT: -flto                   Enable LTO in 'full' mode
! HELP-FC1-NEXT: -fno-analyzed-objects-for-unparse
! HELP-FC1-NEXT:                         Do not use the analyzed objects when unparsing
! HELP-FC1-NEXT: -fno-automatic          Implies the SAVE attribute for non-automatic local objects in subprograms unless RECURSIVE
! HELP-FC1-NEXT: -fno-debug-pass-manager Disables debug printing for the new pass manager
! HELP-FC1-NEXT: -fno-ppc-native-vector-element-order
! HELP-FC1-NEXT:                         Specifies PowerPC non-native vector element order
! HELP-FC1-NEXT: -fno-reformat           Dump the cooked character stream in -E mode
! HELP-FC1-NEXT: -fno-signed-zeros       Allow optimizations that ignore the sign of floating point zeros
! HELP-FC1-NEXT: -fno-stack-arrays       Allocate array temporaries on the heap (default)
! HELP-FC1-NEXT: -fno-version-loops-for-stride
! HELP-FC1-NEXT:                         Do not create unit-strided loops (default)
! HELP-FC1-NEXT: -fopenacc               Enable OpenACC
! HELP-FC1-NEXT: -fopenmp-host-ir-file-path <value>
! HELP-FC1-NEXT:                         Path to the IR file produced by the frontend for the host.
! HELP-FC1-NEXT: -fopenmp-is-target-device
! HELP-FC1-NEXT:                         Generate code only for an OpenMP target device.
! HELP-FC1-NEXT: -fopenmp-target-debug   Enable debugging in the OpenMP offloading device RTL
! HELP-FC1-NEXT: -fopenmp-version=<value>
! HELP-FC1-NEXT:                         Set OpenMP version (e.g. 45 for OpenMP 4.5, 51 for OpenMP 5.1). Default value is 51 for Clang
! HELP-FC1-NEXT: -fopenmp                Parse OpenMP pragmas and generate parallel code.
! HELP-FC1-NEXT: -fpass-plugin=<dsopath> Load pass plugin from a dynamic shared object file (only with new pass manager).
! HELP-FC1-NEXT: -fppc-native-vector-element-order
! HELP-FC1-NEXT:                         Specifies PowerPC native vector element order (default)
! HELP-FC1-NEXT: -freciprocal-math       Allow division operations to be reassociated
! HELP-FC1-NEXT: -fstack-arrays          Attempt to allocate array temporaries on the stack, no matter their size
! HELP-FC1-NEXT: -fsyntax-only           Run the preprocessor, parser and semantic analysis stages
! HELP-FC1-NEXT: -funderscoring          Appends one trailing underscore to external names
! HELP-FC1-NEXT: -fversion-loops-for-stride
! HELP-FC1-NEXT:                         Create unit-strided versions of loops
! HELP-FC1-NEXT: -fxor-operator          Enable .XOR. as a synonym of .NEQV.
! HELP-FC1-NEXT: -help                   Display available options
! HELP-FC1-NEXT: -init-only              Only execute frontend initialization
! HELP-FC1-NEXT: -I <dir>                Add directory to the end of the list of include search paths
! HELP-FC1-NEXT: -load <dsopath>         Load the named plugin (dynamic shared object)
! HELP-FC1-NEXT: -menable-no-infs        Allow optimization to assume there are no infinities.
! HELP-FC1-NEXT: -menable-no-nans        Allow optimization to assume there are no NaNs.
! HELP-FC1-NEXT: -mllvm <value>          Additional arguments to forward to LLVM's option processing
! HELP-FC1-NEXT: -mmlir <value>          Additional arguments to forward to MLIR's option processing
! HELP-FC1-NEXT: -module-dir <dir>       Put MODULE files in <dir>
! HELP-FC1-NEXT: -module-suffix <suffix> Use <suffix> as the suffix for module files (the default value is `.mod`)
! HELP-FC1-NEXT: -mreassociate           Allow reassociation transformations for floating-point instructions
! HELP-FC1-NEXT: -mrelocation-model <value>
! HELP-FC1-NEXT:                         The relocation model to use
! HELP-FC1-NEXT: -mvscale-max=<value>    Specify the vscale maximum. Defaults to the vector length agnostic value of "0". (AArch64/RISC-V only)
! HELP-FC1-NEXT: -mvscale-min=<value>    Specify the vscale minimum. Defaults to "1". (AArch64/RISC-V only)
! HELP-FC1-NEXT: -nocpp                  Disable predefined and command line preprocessor macros
! HELP-FC1-NEXT: -opt-record-file <value>
! HELP-FC1-NEXT:                         File name to use for YAML optimization record output
! HELP-FC1-NEXT: -opt-record-format <value>
! HELP-FC1-NEXT:                         The format used for serializing remarks (default: YAML)
! HELP-FC1-NEXT: -opt-record-passes <value>
! HELP-FC1-NEXT:                         Only record remark information for passes whose names match the given regular expression
! HELP-FC1-NEXT: -o <file>               Write output to <file>
! HELP-FC1-NEXT: -pedantic               Warn on language extensions
! HELP-FC1-NEXT: -pic-is-pie             File is for a position independent executable
! HELP-FC1-NEXT: -pic-level <value>      Value for __PIC__
! HELP-FC1-NEXT: -plugin <name>          Use the named plugin action instead of the default action (use "help" to list available options)
! HELP-FC1-NEXT: -P                      Disable linemarker output in -E mode
! HELP-FC1-NEXT: -Rpass-analysis=<value> Report transformation analysis from optimization passes whose name matches the given POSIX regular expression
! HELP-FC1-NEXT: -Rpass-missed=<value>   Report missed transformations by optimization passes whose name matches the given POSIX regular expression
! HELP-FC1-NEXT: -Rpass=<value>          Report transformations performed by optimization passes whose name matches the given POSIX regular expression
! HELP-FC1-NEXT: -R<remark>              Enable the specified remark
! HELP-FC1-NEXT: -save-temps=<value>     Save intermediate compilation results.
! HELP-FC1-NEXT: -save-temps             Save intermediate compilation results
! HELP-FC1-NEXT: -std=<value>            Language standard to compile for
! HELP-FC1-NEXT: -S                      Only run preprocess and compilation steps
! HELP-FC1-NEXT: -target-cpu <value>     Target a specific cpu type
! HELP-FC1-NEXT: -target-feature <value> Target specific attributes
! HELP-FC1-NEXT: -test-io                Run the InputOuputTest action. Use for development and testing only.
! HELP-FC1-NEXT: -triple <value>         Specify target triple (e.g. i686-apple-darwin9)
! HELP-FC1-NEXT: -U <macro>              Undefine macro <macro>
! HELP-FC1-NEXT: -version                Print the compiler version
! HELP-FC1-NEXT: -W<warning>             Enable the specified warning
! HELP-FC1-NEXT: -x <language>           Treat subsequent input files as having type <language>

! ERROR: error: unknown argument '-helps'; did you mean '-help'
