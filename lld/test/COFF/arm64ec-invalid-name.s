// REQUIRES: aarch64

// Verify that an error is emitted when attempting to export an invalid function name.
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %s -o %t.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %S/Inputs/loadconfig-arm64ec.s -o %t-loadconfig.obj
// RUN: not lld-link -machine:arm64ec -dll -noentry "-export:?func" %t-loadconfig.obj %t.obj 2>&1 | FileCheck %s
// CHECK: error: Invalid ARM64EC function name '?func'

// Verify that we can handle an invalid function name in the archive map.
// RUN: llvm-lib -machine:arm64ec -out:%t.lib %t.obj
// RUN: lld-link -machine:arm64ec -dll -noentry %t-loadconfig.obj %t.lib

        .globl "?func"
"?func":
        ret
