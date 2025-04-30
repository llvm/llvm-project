Flang
=====

Flang (also known as "Classic Flang") is an out-of-tree Fortran compiler
targeting LLVM. It is an open-sourced version of pgfortran, a commercial
Fortran compiler from PGI/NVIDIA. It is different from the new Flang
(formerly known as "F18"; see https://flang.llvm.org/), which has been part
of the LLVM project since 2020, although both are developed by the same
community. It is also unrelated to other projects of the same name, such as
https://github.com/llvm-flang/flang and https://github.com/isanbard/flang.

Classic Flang is used in several downstream commercial projects like the
[AMD](https://developer.amd.com/amd-aocc/), [Arm](https://www.arm.com/products/development-tools/server-and-hpc/allinea-studio/fortran-compiler) and [Huawei](https://support.huaweicloud.com/intl/en-us/ug-bisheng-kunpengdevps/kunpengbisheng_06_0001.html) compilers, and continues to be maintained, but the plan is to replace Classic Flang with the new Flang in the future.

Visit the Flang wiki for more information:

https://github.com/flang-compiler/flang/wiki

To sign up for the developer mailing lists for announcements and discussions,
visit:

https://lists.llvm.org/cgi-bin/mailman/listinfo/flang-dev

We have a flang-compiler channel on Slack. Slack is invitation-only but
anyone can join with the invitation link below:

https://join.slack.com/t/flang-compiler/shared_invite/MjExOTEyMzQ3MjIxLTE0OTk4NzQyNzUtODQzZWEyMjkwYw

## Building Flang

Instructions for building Flang can be found on the Flang wiki:

https://github.com/flang-compiler/flang/wiki/Building-Flang

## Compiler Options

For a list of compiler options, enter:

```
% flang -help
```

Flang accepts all Clang compiler options and supports many, as well as
the following Fortran-specific compiler options:

```lang-none
-noFlangLibs          Do not link against Flang libraries
-mp                   Enable OpenMP and link with with OpenMP library libomp
-nomp                 Do not link with OpenMP library libomp
-Mbackslash           Treat backslash in quoted strings like any other character
-Mnobackslash         Treat backslash in quoted strings like a C-style escape character (Default)
-Mbyteswapio          Swap byte-order for unformatted input/output
-Mfixed               Assume fixed-format source
-Mextend              Allow source lines up to 132 characters
-Mfreeform            Assume free-format source
-Mpreprocess          Run preprocessor for Fortran files
-Mrecursive           Generate code to allow recursive subprograms
-Mstandard            Check standard conformance
-Msave                Assume all variables have SAVE attribute
-module               path to module file (-I also works)
-Mallocatable=95      Select Fortran 95 semantics for assignments to allocatable objects
-Mallocatable=03      Select Fortran 03 semantics for assignments to allocatable objects (Default)
-static-flang-libs    Link using static Flang libraries
-M[no]daz             Treat denormalized numbers as zero
-M[no]flushz          Set SSE to flush-to-zero mode
-Mcache_align         Align large objects on cache-line boundaries
-M[no]fprelaxed       This option is ignored
-fdefault-integer-8   Treat INTEGER and LOGICAL as INTEGER*8 and LOGICAL*8
-fdefault-real-8      Treat REAL as REAL*8
-i8                   Treat INTEGER and LOGICAL as INTEGER*8 and LOGICAL*8
-r8                   Treat REAL as REAL*8
-fno-fortran-main     Don't link in Fortran main
```
