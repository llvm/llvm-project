// RUN: %clang_cc1 -E -fopenacc %s | FileCheck %s --check-prefix=DEFAULT

// DEFAULT: OpenACC:202506:
OpenACC:_OPENACC:

// RUN: %clang_cc1 -E -dM -fopenacc %s | FileCheck %s --check-prefix=MACRO_PRINT_DEF
// MACRO_PRINT_DEF: #define _OPENACC 202506


