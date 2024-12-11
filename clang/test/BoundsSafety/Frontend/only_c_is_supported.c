
// RUN: not %clang -cc1 -fbounds-safety -x c++ %s 2>&1 | FileCheck %s

// RUN: not %clang -cc1 -fbounds-safety -x objective-c %s 2>&1 | FileCheck %s

// RUN: not %clang -cc1 -fbounds-safety -x objective-c++ %s 2>&1 | FileCheck %s

// RUN: not %clang -cc1 -fbounds-safety -x objective-c++ %s 2>&1 | FileCheck %s

// CHECK: error: -fbounds-safety is supported only for C language
