// REQUIRES: aarch64

// RUN: llvm-mc -filetype=obj -triple=aarch64-windows %s -o %t-arm64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %s -o %t-arm64ec.obj
// RUN: rm -f %t.a
// RUN: llvm-ar cr --whole-archive %t.obj %t-arm64.obj %t-arm64ec.obj

// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %S/Inputs/loadconfig-arm64ec.s -o %t-loadconfig-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows %S/Inputs/loadconfig-arm64.s -o %t-loadconfig-arm64.obj

// RUN: lld-link -machine:arm64x -dll -noentry -out:%t.dll %t.obj %t-loadconfig-arm64.obj %t-loadconfig-arm64ec.obj
// RUN: llvm-readobj --coff-exports %t.dll | FileCheck %s

// CHECK:      Format: COFF-ARM64X
// CHECK-NEXT: Arch: aarch64
// CHECK-NEXT: AddressSize: 64bit
// CHECK-NEXT: Export {
// CHECK-NEXT:   Ordinal: 1
// CHECK-NEXT:   Name: sym
// CHECK-NEXT:   RVA: 0x2000
// CHECK-NEXT: }
// CHECK-NEXT: HybridObject {
// CHECK-NEXT:   Format: COFF-ARM64EC
// CHECK-NEXT:   Arch: aarch64
// CHECK-NEXT:   AddressSize: 64bit
// CHECK-NEXT:   Export {
// CHECK-NEXT:     Ordinal: 1
// CHECK-NEXT:     Name: sym
// CHECK-NEXT:     RVA: 0x2004
// CHECK-NEXT:   }
// CHECK-NEXT: }

        .data
        .globl sym
sym:
        .long 0

        .section .drectve
        .ascii "-export:sym"
