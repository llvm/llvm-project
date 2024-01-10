
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

! CHECK:USAGE: flang-new
! CHECK-EMPTY:
! CHECK-NEXT: DRIVER OPTIONS:
! CHECK-NEXT:  --driver-mode=<value> Set the driver mode to either 'gcc', 'g++', 'cpp', 'cl' or 'flang'
! CHECK-EMPTY:
! CHECK-NEXT:OPTIONS:
! CHECK-NEXT: -###                    Print (but do not run) the commands to run for this compilation
! CHECK-NEXT: -ccc-print-phases       Dump list of actions to perform
! CHECK-NEXT: -cpp                    Enable predefined and command line preprocessor macros
! CHECK-NEXT: -c                      Only run preprocess, compile, and assemble steps
! CHECK-NEXT: -dumpmachine            Display the compiler's target processor
! CHECK-NEXT: -dumpversion            Display the version of the compiler
! CHECK-NEXT: -D <macro>=<value>      Define <macro> to <value> (or 1 if <value> omitted)
! CHECK-NEXT: -emit-llvm              Use the LLVM representation for assembler and object files
! CHECK-NEXT: -E                      Only run the preprocessor
! CHECK-NEXT: -falternative-parameter-statement
! CHECK-NEXT:                         Enable the old style PARAMETER statement
! CHECK-NEXT: -fapprox-func           Allow certain math function calls to be replaced with an approximately equivalent calculation
! CHECK-NEXT: -fbackslash             Specify that backslash in string introduces an escape character
! CHECK-NEXT: -fcolor-diagnostics     Enable colors in diagnostics
! CHECK-NEXT: -fconvert=<value>       Set endian conversion of data for unformatted files
! CHECK-NEXT: -fdefault-double-8      Set the default double precision kind to an 8 byte wide type
! CHECK-NEXT: -fdefault-integer-8     Set the default integer and logical kind to an 8 byte wide type
! CHECK-NEXT: -fdefault-real-8        Set the default real kind to an 8 byte wide type
! CHECK-NEXT: -ffast-math             Allow aggressive, lossy floating-point optimizations
! CHECK-NEXT: -ffixed-form            Process source files in fixed form
! CHECK-NEXT: -ffixed-line-length=<value>
! CHECK-NEXT:                         Use <value> as character line width in fixed mode
! CHECK-NEXT: -ffp-contract=<value>   Form fused FP ops (e.g. FMAs)
! CHECK-NEXT: -ffree-form             Process source files in free form
! CHECK-NEXT: -fhonor-infinities      Specify that floating-point optimizations are not allowed that assume arguments and results are not +-inf.
! CHECK-NEXT: -fhonor-nans            Specify that floating-point optimizations are not allowed that assume arguments and results are not NANs.
! CHECK-NEXT: -fimplicit-none         No implicit typing allowed unless overridden by IMPLICIT statements
! CHECK-NEXT: -finput-charset=<value> Specify the default character set for source files
! CHECK-NEXT: -fintegrated-as         Enable the integrated assembler
! CHECK-NEXT: -fintrinsic-modules-path <dir>
! CHECK-NEXT:                         Specify where to find the compiled intrinsic modules
! CHECK-NEXT: -flang-deprecated-no-hlfir
! CHECK-NEXT:                         Do not use HLFIR lowering (deprecated)
! CHECK-NEXT: -flang-experimental-hlfir
! CHECK-NEXT:                         Use HLFIR lowering (experimental)
! CHECK-NEXT: -flang-experimental-polymorphism
! CHECK-NEXT:                         Enable Fortran 2003 polymorphism (experimental)
! CHECK-NEXT: -flarge-sizes           Use INTEGER(KIND=8) for the result type in size-related intrinsics
! CHECK-NEXT: -flogical-abbreviations Enable logical abbreviations
! CHECK-NEXT: -flto=auto              Enable LTO in 'full' mode
! CHECK-NEXT: -flto=jobserver         Enable LTO in 'full' mode
! CHECK-NEXT: -flto=<value>           Set LTO mode
! CHECK-NEXT: -flto                   Enable LTO in 'full' mode
! CHECK-NEXT: -fms-runtime-lib=<value>
! CHECK-NEXT:                         Select Windows run-time library
! CHECK-NEXT: -fno-automatic          Implies the SAVE attribute for non-automatic local objects in subprograms unless RECURSIVE
! CHECK-NEXT: -fno-color-diagnostics  Disable colors in diagnostics
! CHECK-NEXT: -fno-fortran-main       Do not include Fortran_main.a (provided by Flang) when linking
! CHECK-NEXT: -fno-integrated-as      Disable the integrated assembler
! CHECK-NEXT: -fno-lto                Disable LTO mode (default)
! CHECK-NEXT: -fno-ppc-native-vector-element-order
! CHECK-NEXT:                         Specifies PowerPC non-native vector element order
! CHECK-NEXT: -fno-signed-zeros       Allow optimizations that ignore the sign of floating point zeros
! CHECK-NEXT: -fno-stack-arrays       Allocate array temporaries on the heap (default)
! CHECK-NEXT: -fno-version-loops-for-stride
! CHECK-NEXT:                         Do not create unit-strided loops (default)
! CHECK-NEXT: -fomit-frame-pointer    Omit the frame pointer from functions that don't need it. Some stack unwinding cases, such as profilers and sanitizers, may prefer specifying -fno-omit-frame-pointer. On many targets, -O1 and higher omit the frame pointer by default. -m[no-]omit-leaf-frame-pointer takes precedence for leaf functions
! CHECK-NEXT: -fopenacc               Enable OpenACC
! CHECK-NEXT: -fopenmp-assume-no-nested-parallelism
! CHECK-NEXT:                         Assert no nested parallel regions in the GPU
! CHECK-NEXT: -fopenmp-assume-no-thread-state
! CHECK-NEXT:                         Assert no thread in a parallel region modifies an ICV
! CHECK-NEXT: -fopenmp-target-debug   Enable debugging in the OpenMP offloading device RTL
! CHECK-NEXT: -fopenmp-targets=<value>
! CHECK-NEXT:                         Specify comma-separated list of triples OpenMP offloading targets to be supported
! CHECK-NEXT: -fopenmp-version=<value>
! CHECK-NEXT:                         Set OpenMP version (e.g. 45 for OpenMP 4.5, 51 for OpenMP 5.1). Default value is 51 for Clang
! CHECK-NEXT: -fopenmp                Parse OpenMP pragmas and generate parallel code.
! CHECK-NEXT: -foptimization-record-file=<file>
! CHECK-NEXT:                         Specify the output name of the file containing the optimization remarks. Implies -fsave-optimization-record. On Darwin platforms, this cannot be used with multiple -arch <arch> options.
! CHECK-NEXT: -foptimization-record-passes=<regex>
! CHECK-NEXT:                         Only include passes which match a specified regular expression in the generated optimization record (by default, include all passes)
! CHECK-NEXT: -fpass-plugin=<dsopath> Load pass plugin from a dynamic shared object file (only with new pass manager).
! CHECK-NEXT: -fppc-native-vector-element-order
! CHECK-NEXT:                         Specifies PowerPC native vector element order (default)
! CHECK-NEXT: -freciprocal-math       Allow division operations to be reassociated
! CHECK-NEXT: -fropi                  Generate read-only position independent code (ARM only)
! CHECK-NEXT: -frwpi                  Generate read-write position independent code (ARM only)
! CHECK-NEXT: -fsave-optimization-record=<format>
! CHECK-NEXT:                         Generate an optimization record file in a specific format
! CHECK-NEXT: -fsave-optimization-record
! CHECK-NEXT:                         Generate a YAML optimization record file
! CHECK-NEXT: -fstack-arrays          Attempt to allocate array temporaries on the stack, no matter their size
! CHECK-NEXT: -fsyntax-only           Run the preprocessor, parser and semantic analysis stages
! CHECK-NEXT: -funderscoring          Appends one trailing underscore to external names
! CHECK-NEXT: -fveclib=<value>        Use the given vector functions library
! CHECK-NEXT: -fversion-loops-for-stride
! CHECK-NEXT:                         Create unit-strided versions of loops
! CHECK-NEXT: -fxor-operator          Enable .XOR. as a synonym of .NEQV.
! CHECK-NEXT: -gline-directives-only  Emit debug line info directives only
! CHECK-NEXT: -gline-tables-only      Emit debug line number tables only
! CHECK-NEXT: -gpulibc                Link the LLVM C Library for GPUs
! CHECK-NEXT: -g                      Generate source-level debug information
! CHECK-NEXT: --help-hidden           Display help for hidden options
! CHECK-NEXT: -help                   Display available options
! CHECK-NEXT: -I <dir>                Add directory to the end of the list of include search paths
! CHECK-NEXT: -L <dir>                Add directory to library search path
! CHECK-NEXT: -march=<value>          For a list of available architectures for the target use '-mcpu=help'
! CHECK-NEXT: -mcode-object-version=<value>
! CHECK-NEXT:                         Specify code object ABI version. Defaults to 4. (AMDGPU only)
! CHECK-NEXT: -mcpu=<value>           For a list of available CPUs for the target use '-mcpu=help'
! CHECK-NEXT: -mllvm=<arg>            Alias for -mllvm
! CHECK-NEXT: -mllvm <value>          Additional arguments to forward to LLVM's option processing
! CHECK-NEXT: -mmlir <value>          Additional arguments to forward to MLIR's option processing
! CHECK-NEXT: -module-dir <dir>       Put MODULE files in <dir>
! CHECK-NEXT: -mrvv-vector-bits=<value>
! CHECK-NEXT:                         Specify the size in bits of an RVV vector register
! CHECK-NEXT: -msve-vector-bits=<value>
! CHECK-NEXT:                          Specify the size in bits of an SVE vector register. Defaults to the vector length agnostic value of "scalable". (AArch64 only)
! CHECK-NEXT: --no-offload-arch=<value>
! CHECK-NEXT:                         Remove CUDA/HIP offloading device architecture (e.g. sm_35, gfx906) from the list of devices to compile for. 'all' resets the list to its default value.
! CHECK-NEXT: -nocpp                  Disable predefined and command line preprocessor macros
! CHECK-NEXT: -nogpulib               Do not link device library for CUDA/HIP device compilation
! CHECK-NEXT: --offload-arch=<value>  Specify an offloading device architecture for CUDA, HIP, or OpenMP. (e.g. sm_35). If 'native' is used the compiler will detect locally installed architectures. For HIP offloading, the device architecture can be followed by target ID features delimited by a colon (e.g. gfx908:xnack+:sramecc-). May be specified more than once.
! CHECK-NEXT: --offload-device-only   Only compile for the offloading device.
! CHECK-NEXT: --offload-host-device   Compile for both the offloading host and device (default).
! CHECK-NEXT: --offload-host-only     Only compile for the offloading host.
! CHECK-NEXT: -o <file>               Write output to <file>
! CHECK-NEXT: -pedantic               Warn on language extensions
! CHECK-NEXT: -print-effective-triple Print the effective target triple
! CHECK-NEXT: -print-target-triple    Print the normalized target triple
! CHECK-NEXT: -P                      Disable linemarker output in -E mode
! CHECK-NEXT: -Rpass-analysis=<value> Report transformation analysis from optimization passes whose name matches the given POSIX regular expression
! CHECK-NEXT: -Rpass-missed=<value>   Report missed transformations by optimization passes whose name matches the given POSIX regular expression
! CHECK-NEXT: -Rpass=<value>          Report transformations performed by optimization passes whose name matches the given POSIX regular expression
! CHECK-NEXT: -R<remark>              Enable the specified remark
! CHECK-NEXT: -save-temps=<value>     Save intermediate compilation results.
! CHECK-NEXT: -save-temps             Save intermediate compilation results
! CHECK-NEXT: -std=<value>            Language standard to compile for
! CHECK-NEXT: -S                      Only run preprocess and compilation steps
! CHECK-NEXT: --target=<value>        Generate code for the given target
! CHECK-NEXT: -U <macro>              Undefine macro <macro>
! CHECK-NEXT: --version               Print version information
! CHECK-NEXT: -v                      Show commands to run and use verbose output
! CHECK-NEXT: -Wl,<arg>               Pass the comma separated arguments in <arg> to the linker
! CHECK-NEXT: -W<warning>             Enable the specified warning
! CHECK-NEXT: -Xflang <arg>           Pass <arg> to the flang compiler
! CHECK-NEXT: -x <language>           Treat subsequent input files as having type <language>


! ERROR-FLANG: error: unknown argument '-help-hidden'; did you mean '--help-hidden'?

! Frontend driver -help-hidden is not supported
! ERROR-FLANG-FC1: error: unknown argument: '{{.*}}'
