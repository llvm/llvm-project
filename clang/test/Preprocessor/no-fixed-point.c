/// Assert the fixed point precision macros according to ISO/IEC TR 18037:2008 7.18a.3 are not
/// defined when -ffixed-point is not provided.

// RUN: %clang_cc1 -triple=x86_64 -E -dM -x c < /dev/null | FileCheck -match-full-lines %s
// RUN: %clang_cc1 -triple=x86_64 -E -dM -x c++ < /dev/null | FileCheck -match-full-lines %s

// CHECK-NOT:#define __SFRACT_FBIT__ 7
