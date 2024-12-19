// The test doesn't run in a PTY, so "auto" defaults to off.
// RUN: %clang_dxc -Tlib_6_7 -fdiagnostics-color=auto -### -- %s 2>&1 | FileCheck -check-prefix=NO_COLOR %s

// RUN: %clang_dxc -Tlib_6_7 -fdiagnostics-color -### %s 2>&1 | FileCheck -check-prefix=COLOR %s
// RUN: %clang_dxc -Tlib_6_7 -fdiagnostics-color=always -### %s 2>&1 | FileCheck -check-prefix=COLOR %s
// RUN: %clang_dxc -Tlib_6_7 -fdiagnostics-color=never -### %s 2>&1 | FileCheck -check-prefix=NO_COLOR %s
// COLOR: "-fcolor-diagnostics"
// NO_COLOR-NOT: "-fcolor-diagnostics"