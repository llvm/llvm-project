// RUN: %clang_cc1 -E -dM -triple i586-intel-elfiamcu | FileCheck %s

// CHECK: #define __iamcu
// CHECK: #define __iamcu__

