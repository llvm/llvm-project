// RUN: not %clang -fbounds-safety-experimental -x c++ %s 2>&1 | FileCheck %s

// RUN: not %clang -fbounds-safety-experimental -x objective-c %s 2>&1 | FileCheck %s

// RUN: not %clang -fbounds-safety-experimental -x objective-c++ %s 2>&1 | FileCheck %s

// RUN: not %clang -cc1 -fbounds-safety-experimental -x c++ %s 2>&1 | FileCheck %s

// RUN: not %clang -cc1 -fbounds-safety-experimental -x objective-c %s 2>&1 | FileCheck %s

// RUN: not %clang -cc1 -fbounds-safety-experimental -x objective-c++ %s 2>&1 | FileCheck %s

// RUN: not %clang -cc1 -fbounds-safety-experimental -x objective-c++ %s 2>&1 | FileCheck %s

// CHECK: error: bounds safety is only supported for C
