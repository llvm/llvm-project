// RUN: %clang_cc1 -E -fopenacc %s | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang_cc1 -E -fopenacc -fexperimental-openacc-macro-override 202211 %s | FileCheck %s --check-prefix=OVERRIDE

// DEFAULT: OpenACC:1:
// OVERRIDE: OpenACC:202211:
OpenACC:_OPENACC:

// RUN: %clang_cc1 -E -dM -fopenacc %s | FileCheck %s --check-prefix=MACRO_PRINT_DEF
// RUN: %clang_cc1 -E -dM -fopenacc -fexperimental-openacc-macro-override 202211 %s | FileCheck %s --check-prefix=MACRO_PRINT_OVR
// MACRO_PRINT_DEF: #define _OPENACC 1
// MACRO_PRINT_OVR: #define _OPENACC 202211


