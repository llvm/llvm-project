// This test verifies that the correct macros are predefined.
//
// RUN: %clang_cc1 %s -E -dM -triple i686-pc-win32 -fms-extensions -fms-compatibility \
// RUN:     -fmsc-version=1300 -o - | FileCheck %s --check-prefix=CHECK-MS
// CHECK-MS: #define _INTEGRAL_MAX_BITS 64
// CHECK-MS: #define _MSC_EXTENSIONS 1
// CHECK-MS: #define _MSC_VER 1300
// CHECK-MS: #define _M_IX86 600
// CHECK-MS: #define _M_IX86_FP
// CHECK-MS: #define _WIN32 1
// CHECK-MS-NOT: #define __GNUC__
//
// RUN: %clang_cc1 %s -E -dM -ffast-math -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-FAST-MATH
// CHECK-FAST-MATH: #define __FAST_MATH__
// CHECK-FAST-MATH: #define __FINITE_MATH_ONLY__ 1
//
// RUN: %clang_cc1 %s -E -dM -ffinite-math-only -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-FINITE-MATH-ONLY
// CHECK-FINITE-MATH-ONLY: #define __FINITE_MATH_ONLY__ 1
//
// RUN: %clang %s -E -dM -fno-finite-math-only -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-FINITE-MATH-ONLY
// CHECK-NO-FINITE-MATH-ONLY: #define __FINITE_MATH_ONLY__ 0
//
// RUN: %clang_cc1 %s -E -dM -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-FINITE-MATH-FLAG-UNDEFINED
// CHECK-FINITE-MATH-FLAG-UNDEFINED: #define __FINITE_MATH_ONLY__ 0
