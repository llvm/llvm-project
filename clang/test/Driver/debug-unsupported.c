// RUN: %clang -c -gstabs %s 2>&1 | FileCheck %s
// RUN: %clang -c -gstabs+ %s 2>&1 | FileCheck %s
// RUN: %clang -c -gcoff %s 2>&1 | FileCheck %s
// RUN: %clang -c -gxcoff %s 2>&1 | FileCheck %s
// RUN: %clang -c -gxcoff+ %s 2>&1 | FileCheck %s
// RUN: %clang -c -gvms %s 2>&1 | FileCheck %s
// RUN: %clang -c -gstabs1 %s 2>&1 | FileCheck %s
// RUN: %clang -c -gcoff2 %s 2>&1 | FileCheck %s
// RUN: %clang -c -gxcoff3 %s 2>&1 | FileCheck %s
// RUN: %clang -c -gvms0 %s 2>&1 | FileCheck %s
// RUN: %clang -c -gtoggle %s 2>&1 | FileCheck %s
//
// CHECK: error: unsupported option
