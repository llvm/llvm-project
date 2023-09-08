// RUN: %clang_dxc -Tlib_6_7 -### -g %s 2>&1 | FileCheck %s
// RUN: %clang_dxc -Tlib_6_7 -### /Zi %s 2>&1 | FileCheck %s
// RUN: %clang_dxc -Tlib_6_7 -### /Zi /Qembed_debug %s 2>&1 | FileCheck %s
// RUN: %clang_dxc -Tlib_6_7 -### -Zi %s 2>&1 | FileCheck %s
// RUN: %clang_dxc -Tlib_6_7 -### -Zi -Qembed_debug %s 2>&1 | FileCheck %s
// RUN: %clang_dxc -Tlib_6_7 -### -Zi -gcodeview %s 2>&1 | FileCheck %s -check-prefixes=CHECK,CHECK-CV
// RUN: %clang_dxc -Tlib_6_7 -### -Zi -gdwarf %s 2>&1 | FileCheck %s -check-prefixes=CHECK,CHECK-DWARF
// RUN: %clang_dxc -Tlib_6_7 -### -gcodeview -Zi %s 2>&1 | FileCheck %s -check-prefixes=CHECK,CHECK-CV
// RUN: %clang_dxc -Tlib_6_7 -### -gdwarf -Zi %s 2>&1 | FileCheck %s -check-prefixes=CHECK,CHECK-DWARF

// CHECK: "-cc1"
// CHECK-CV-SAME: -gcodeview
// CHECK-SAME: "-debug-info-kind=constructor"
// CHECK-DWARF-SAME: -dwarf-version
