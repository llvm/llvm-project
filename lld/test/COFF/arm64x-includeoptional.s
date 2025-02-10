// REQUIRES: aarch64

// RUN: llvm-mc -filetype=obj -triple=aarch64-windows %s -o %t.arm64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %s -o %t.arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %S/Inputs/loadconfig-arm64ec.s -o %t-loadconfig-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows %S/Inputs/loadconfig-arm64.s -o %t-loadconfig-arm64.obj

// RUN: llvm-lib -machine:arm64x -out:%t-test.lib %t.arm64.obj %t.arm64ec.obj %t-loadconfig-arm64ec.obj %t-loadconfig-arm64.obj
// RUN: lld-link -machine:arm64x -dll -noentry -out:%t.dll %t-test.lib -includeoptional:sym

// RUN: llvm-readobj --hex-dump=.test %t.dll | FileCheck %s
// CHECK: 0x180004000 01000000 01000000

        .globl sym
        .section .test,"dr"
sym:
        .word 1
