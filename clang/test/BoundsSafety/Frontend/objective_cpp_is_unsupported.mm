

// RUN: not %clang_cc1 -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -fbounds-attributes %s 2>&1 | FileCheck %s

// CHECK: error: -fbounds-safety is supported only for C language
