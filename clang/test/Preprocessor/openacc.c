// RUN: %clang_cc1 -E -fopenacc %s | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang_cc1 -E -fopenacc -fexperimental-openacc-macro-override 202211 %s | FileCheck %s --check-prefix=OVERRIDE

// DEFAULT: OpenACC:1:
// OVERRIDE: OpenACC:202211:
OpenACC:_OPENACC:
