// REQUIRES: aarch64
// RUN: split-file %s %t.dir && cd %t.dir

// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows test.s -o test-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows test.s -o test-arm64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows other.s -o other-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows other.s -o other-arm64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %S/Inputs/loadconfig-arm64ec.s -o loadconfig-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows %S/Inputs/loadconfig-arm64.s -o loadconfig-arm64.obj

// RUN: lld-link -machine:arm64x -dll -noentry test-arm64.obj test-arm64ec.obj other-arm64.obj other-arm64ec.obj \
// RUN:          loadconfig-arm64.obj loadconfig-arm64ec.obj -out:out.dll -wrap:sym -wrap:nosuchsym

// RUN: llvm-readobj --hex-dump=.test out.dll | FileCheck %s
// CHECK: 0x180004000 02000000 02000000 01000000 02000000
// CHECK: 0x180004010 02000000 01000000

#--- test.s
        .section .test,"dr"
        .word sym
        .word __wrap_sym
        .word __real_sym

#--- other.s
        .global sym
        .global __wrap_sym
        .global __real_sym

sym = 1
__wrap_sym = 2
__real_sym = 3
