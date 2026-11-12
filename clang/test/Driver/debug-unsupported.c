// RUN: not %clang -c -gstabs %s 2>&1 | FileCheck %s
// RUN: not %clang -c -gstabs+ %s 2>&1 | FileCheck %s
// RUN: not %clang -c -gcoff %s 2>&1 | FileCheck %s
// RUN: not %clang -c -gxcoff %s 2>&1 | FileCheck %s
// RUN: not %clang -c -gxcoff+ %s 2>&1 | FileCheck %s
// RUN: not %clang -c -gvms %s 2>&1 | FileCheck %s
// RUN: not %clang -c -gstabs1 %s 2>&1 | FileCheck %s
// RUN: not %clang -c -gcoff2 %s 2>&1 | FileCheck %s
// RUN: not %clang -c -gxcoff3 %s 2>&1 | FileCheck %s
// RUN: not %clang -c -gvms0 %s 2>&1 | FileCheck %s
// RUN: not %clang -c -gtoggle %s 2>&1 | FileCheck %s
//
// CHECK: error: unsupported option
