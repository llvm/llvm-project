// Check that debug-info-related parts are emitted when clang-dxc is invoked with debug option.

// REQUIRES: directx-registered-target
// RUN: %clang_dxc -Tlib_6_7 /Fo %t.dxbc -g %s 2>&1
// RUN: obj2yaml %t.dxbc | FileCheck %s
// RUN: %clang_dxc -Tlib_6_7 /Fo %t.dxbc /Zi %s 2>&1
// RUN: obj2yaml %t.dxbc | FileCheck %s
// RUN: %clang_dxc -Tlib_6_7 /Fo %t.dxbc /Zi /Qembed_debug %s 2>&1
// RUN: obj2yaml %t.dxbc | FileCheck %s
// RUN: %clang_dxc -Tlib_6_7 /Fo %t.dxbc -Zi %s 2>&1
// RUN: obj2yaml %t.dxbc | FileCheck %s
// RUN: %clang_dxc -Tlib_6_7 /Fo %t.dxbc -Zi -Qembed_debug %s 2>&1
// RUN: obj2yaml %t.dxbc | FileCheck %s
// RUN: %clang_dxc -Tlib_6_7 /Fo %t.dxbc -Zi -gcodeview %s 2>&1
// RUN: obj2yaml %t.dxbc | FileCheck %s
// RUN: %clang_dxc -Tlib_6_7 /Fo %t.dxbc -Zi -gdwarf %s 2>&1
// RUN: obj2yaml %t.dxbc | FileCheck %s
// RUN: %clang_dxc -Tlib_6_7 /Fo %t.dxbc -gcodeview -Zi %s 2>&1
// RUN: obj2yaml %t.dxbc | FileCheck %s
// RUN: %clang_dxc -Tlib_6_7 /Fo %t.dxbc -gdwarf -Zi %s 2>&1
// RUN: obj2yaml %t.dxbc | FileCheck %s

// CHECK: - Name: ILDN

// Check that /Qpdb_in_private emits a PRIV part in the output container.
// RUN: %clang_dxc -Tlib_6_7 /Fo %t-priv.dxbc /Zi /Qpdb_in_private %s 2>&1
// RUN: obj2yaml %t-priv.dxbc | FileCheck %s --check-prefix=CHECK-PRIV

// Without /Qpdb_in_private, PRIV is not emitted.
// RUN: %clang_dxc -Tlib_6_7 /Fo %t-no-priv.dxbc /Zi %s 2>&1
// RUN: obj2yaml %t-no-priv.dxbc | FileCheck %s --check-prefix=CHECK-NO-PRIV

// CHECK-PRIV: - Name: PRIV
// CHECK-NO-PRIV-NOT: - Name: PRIV

[numthreads(1, 1, 1)] void main() {}
