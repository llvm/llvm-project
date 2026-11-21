// RUN: %clang_dxc -Tlib_6_7 -### -g %s 2>&1 | FileCheck %s --check-prefix=CHECK,CHECK-CMD
// RUN: %clang_dxc -Tlib_6_7 -### /Zi %s 2>&1 | FileCheck %s
// RUN: %clang_dxc -Tlib_6_7 -### /Zi /Qembed_debug %s 2>&1 | FileCheck %s
// RUN: %clang_dxc -Tlib_6_7 -### -Zi %s 2>&1 | FileCheck %s
// RUN: %clang_dxc -Tlib_6_7 -### -Zi -Qembed_debug %s 2>&1 | FileCheck %s
// RUN: %clang_dxc -Tlib_6_7 -### /Zi /Qpdb_in_private %s 2>&1 | FileCheck %s -check-prefixes=CHECK,CHECK-PRIV
// RUN: %clang_dxc -Tlib_6_7 -### -Zi -Zss %s 2>&1 | FileCheck %s --check-prefix=CHECK,CHECK-ZSS
// RUN: %clang_dxc -Tlib_6_7 -### /Zi -Qsource_in_debug_module %s 2>&1 | FileCheck %s --check-prefix=CHECK,CHECK-SIDM
// RUN: %clang_dxc -Tlib_6_7 -### -Zi -Qstrip_debug %s 2>&1 | FileCheck %s --check-prefix=CHECK,CHECK-STRIP
// RUN: %clang_dxc -Tlib_6_7 -### -Zi -gcodeview %s 2>&1 | FileCheck %s -check-prefixes=CHECK,CHECK-CV
// RUN: %clang_dxc -Tlib_6_7 -### -Zi -gdwarf %s 2>&1 | FileCheck %s -check-prefixes=CHECK,CHECK-DWARF
// RUN: %clang_dxc -Tlib_6_7 -### -gcodeview -Zi %s 2>&1 | FileCheck %s -check-prefixes=CHECK,CHECK-CV
// RUN: %clang_dxc -Tlib_6_7 -### -gdwarf -Zi %s 2>&1 | FileCheck %s -check-prefixes=CHECK,CHECK-DWARF

// CHECK: "-cc1"
// CHECK-CV-SAME: -gcodeview
// CHECK-SAME: "-debug-info-kind=constructor"
// Make sure dwarf-version is 4.
// CHECK-DWARF-SAME: -dwarf-version=4
// Check that the flags are converted to their llc equivalents.
// CHECK-PRIV-SAME: --dx-pdb-in-private
// CHECK-ZSS-SAME: -dx-Zss
// CHECK-SIDM-SAME: --dx-source-in-debug-module
// CHECK-STRIP-SAME: --dx-strip-debug
// Make sure dxc command line arguments are passed to clang invocation.
// CHECK-SAME: -fdx-record-command-line
// CHECK-CMD-SAME: --driver-mode=dxc -T lib_6_7 -### -g {{.*}}dxc_debug.hlsl

// Check errors and warnings
// RUN: not %clang_dxc -Tlib_6_7 -### -Zss -Zsb %s 2>&1 | FileCheck %s --check-prefix=CHECK-ZSS-ZSB
// CHECK-ZSS-ZSB: cannot specify both /Zss and /Zsb
