// RUN: rm -rf %t
// RUN: mkdir %t

// Warn about linker options being ignored when not linking
// RUN: %clang %s -lfoo -o %t/tmp1.pch -### 2>&1 | FileCheck %s --check-prefix=UNUSED-L,SINGLEHEADER
// RUN: %clang %s -x c++-header -lfoo -o %t/tmp1.pch -### 2>&1 | FileCheck %s --check-prefix=UNUSED-L,SINGLEHEADER
// UNUSED-L: clang: warning: -lfoo: 'linker' input unused [-Wunused-command-line-argument]

// RUN: %clang %s -Wl,--whole-archive -o %t/tmp1.pch -### 2>&1 | FileCheck %s --check-prefix=UNUSED-WL,SINGLEHEADER
// UNUSED-WL: clang: warning: -Wl,--whole-archive: 'linker' input unused [-Wunused-command-line-argument]

// RUN: %clang %S/Inputs/header1.h %S/Inputs/header2.h -lfoo -### 2>&1 | FileCheck %s --check-prefix=UNUSED-L,MULTIHEADER


// Error with single -o when there are multiple output files
// RUN: not %clang %S/Inputs/header1.h %S/Inputs/header2.h -lfoo -o %t/tmp2.pch -### 2>&1 | FileCheck %s --check-prefix=UNUSED-L,MULTIOUTPUT
// MULTIOUTPUT: clang: error: cannot specify -o when generating multiple output files

// An actual linker input file (object0.o) triggers an error, not a warning
// RUN: not %clang %s %S/Inputs/object0.o -o %t/tmp3.pch -### 2>&1 | FileCheck %s --check-prefix=MULTIOUTPUT


// Normal case: Single header file input compiles to .pch even without --precompile
// RUN: %clang %s -o %t/tmp1.pch -### 2>&1 | FileCheck %s --check-prefix=SINGLEHEADER
// SINGLEHEADER: "-cc1"
// SINGLEHEADER: "-emit-pch"
// SINGLEHEADER: "-o"
// SINGLEHEADER: tmp1.pch


// Multiple header files input compiles to one .pch each even without --precompile
// RUN: %clang %S/Inputs/header1.h %S/Inputs/header2.h -### 2>&1 | FileCheck %s --check-prefix=MULTIHEADER
// MULTIHEADER: "-cc1"
// MULTIHEADER: -emit-pch
// MULTIHEADER: "-o"
// MULTIHEADER: header1.h.pch"
// MULTIHEADER: "-cc1"
// MULTIHEADER: -emit-pch
// MULTIHEADER: "-o"
// MULTIHEADER: header2.h.pch"
