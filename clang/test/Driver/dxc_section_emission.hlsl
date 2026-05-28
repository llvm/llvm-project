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

[numthreads(1, 1, 1)] void main() {}
