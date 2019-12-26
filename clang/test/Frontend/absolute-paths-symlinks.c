// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: cd %t
// RUN: cp %s test.c
// RUN: ln -sf test.c link.c
// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-absolute-paths link.c 2>&1 | FileCheck %s

// Verify that -fdiagnostics-absolute-paths resolve symbolic links in
// diagnostics messages.

// CHECK: test.c
// CHECK-SAME: error: unknown type name
This do not compile

// REQUIRES: shell
// Don't make symlinks on Windows.
// UNSUPPORTED: system-windows
