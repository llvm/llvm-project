// RUN: %clang_cl /TC /dev/null /E -Xclang -dM 2> /dev/null | FileCheck -match-full-lines %s --check-prefix=NOSTDC
// RUN: %clang_cl /TC /dev/null /E -Xclang -dM /Zc:__STDC__ 2> /dev/null | FileCheck -match-full-lines %s --check-prefix=YESSTDC
// __STDC__ should never be defined in C++ mode with fms-compatibility.
// RUN: %clang_cl /dev/null /E -Xclang -dM 2>&1 | FileCheck %s --check-prefix=NOSTDC
// RUN: %clang_cl /dev/null /E -Xclang -dM /Zc:__STDC__ 2>&1 | FileCheck %s --check-prefix=ZCSTDCIGNORED
// YESSTDC: #define __STDC__ 1
// NOSTDC-NOT: #define __STDC__ 1
// ZCSTDCIGNORED-NOT: #define __STDC__ 1
// ZCSTDCIGNORED: argument unused during compilation
