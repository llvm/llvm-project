// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows %s -o %t.obj
// RUN: not lld-link -entry:_start -subsystem:console %t.obj -out:%t.exe -dynamicbase:no 2>&1 | FileCheck %s

// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %s -o %t.obj
// RUN: not lld-link -entry:_start -subsystem:console %t.obj -out:%t.exe -dynamicbase:no -machine:arm64ec 2>&1 \
// RUN:              | FileCheck %s -check-prefix=ARM64EC
// RUN: not lld-link -entry:_start -subsystem:console %t.obj -out:%t.exe -dynamicbase:no -machine:arm64x -dll -noentry 2>&1 \
// RUN:              | FileCheck %s -check-prefix=ARM64X
 .globl _start
_start:
 ret

# CHECK: dynamicbase:no is not compatible with arm64
# ARM64EC: dynamicbase:no is not compatible with arm64ec
# ARM64X: dynamicbase:no is not compatible with arm64x
