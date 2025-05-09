// REQUIRES: aarch64, x86
// RUN: split-file %s %t.dir && cd %t.dir

// RUN: llvm-mc -filetype=obj -triple=aarch64-windows crt1-arm64.s -o crt1-arm64.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows crt2-arm64.s -o crt2-arm64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows crt1-arm64ec.s -o crt1-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=x86_64-windows crt2-amd64.s -o crt2-amd64.obj

// Check that .CRT chunks are correctly sorted and that EC and native chunks are split.

// RUN: lld-link -out:out.dll -machine:arm64x -dll -noentry crt1-arm64.obj crt2-arm64.obj crt1-arm64ec.obj crt2-amd64.obj
// RUN: llvm-readobj --hex-dump=.CRT out.dll | FileCheck %s

// RUN: lld-link -out:out2.dll -machine:arm64x -dll -noentry crt1-arm64.obj crt1-arm64ec.obj crt2-arm64.obj crt2-amd64.obj
// RUN: llvm-readobj --hex-dump=.CRT out2.dll | FileCheck %s

// RUN: lld-link -out:out3.dll -machine:arm64x -dll -noentry crt2-amd64.obj crt1-arm64ec.obj crt2-arm64.obj crt1-arm64.obj
// RUN: llvm-readobj --hex-dump=.CRT out3.dll | FileCheck %s

// CHECK:      0x180002000 01000000 00000000 02000000 00000000
// CHECK-NEXT: 0x180002010 03000000 00000000 11000000 00000000
// CHECK-NEXT: 0x180002020 12000000 00000000 13000000 00000000

#--- crt1-arm64.s
        .section .CRT$A,"dr"
        .xword 1
        .section .CRT$Z,"dr"
        .xword 3

#--- crt2-arm64.s
        .section .CRT$B,"dr"
        .xword 2

#--- crt1-arm64ec.s
        .section .CRT$A,"dr"
        .xword 0x11
        .section .CRT$Z,"dr"
        .xword 0x13

#--- crt2-amd64.s
        .section .CRT$B,"dr"
        .quad 0x12
