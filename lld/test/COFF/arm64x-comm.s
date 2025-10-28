// REQUIRES: aarch64

// Check that -aligncomm applies to both native and EC symbols.

// RUN: llvm-mc -filetype=obj -triple=aarch64-windows-gnu %s -o %t-arm64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows-gnu %s -o %t-arm64ec.obj
// RUN: lld-link -machine:arm64x -lldmingw -dll -noentry -out:%t.dll %t-arm64.obj %t-arm64ec.obj
// RUN: llvm-readobj --hex-dump=.test %t.dll | FileCheck %s
// CHECK: 0x180004000 10200000 18200000 20200000 28200000

// RUN: lld-link -machine:arm64ec -lldmingw -dll -noentry -out:%t-ec.dll %t-arm64.obj %t-arm64ec.obj
// RUN: llvm-readobj --hex-dump=.test %t-ec.dll | FileCheck %s

        .data
        .word 0

        .section .test,"dr"
        .rva sym
        .rva sym2
        .comm sym,4,4
        .comm sym2,4,3
