// REQUIRES: aarch64
// RUN: split-file %s %t.dir && cd %t.dir

// RUN: llvm-mc -filetype=obj -triple=aarch64-windows arm64-data-sym.s -o arm64-data-sym.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows arm64ec-data-sym.s -o arm64ec-data-sym.obj
// RUN: lld-link -machine:arm64x -dll -out:out.dll -map -mapinfo:exports arm64-data-sym.obj arm64ec-data-sym.obj
// RUN: FileCheck %s < out.map

// CHECK:      Start         Length     Name                   Class
// CHECK-NEXT: 0001:00000000 00001004H .text                   CODE
// CHECK-NEXT: 0004:00000000 00000008H .data                   DATA
// CHECK-NEXT: 0004:00000008 00000000H .bss                    DATA
// CHECK-EMPTY:
// CHECK-NEXT:  Address         Publics by Value              Rva+Base               Lib:Object
// CHECK-EMPTY:
// CHECK-NEXT: 0001:00000000       _DllMainCRTStartup         0000000180001000     arm64-data-sym.obj
// CHECK-NEXT: 0001:00001000       _DllMainCRTStartup         0000000180002000     arm64ec-data-sym.obj
// CHECK-NEXT: 0004:00000000       arm64_data_sym             0000000180005000     arm64-data-sym.obj
// CHECK-NEXT: 0004:00000004       arm64ec_data_sym           0000000180005004     arm64ec-data-sym.obj
// CHECK-EMPTY:
// CHECK-NEXT: entry point at         0002:00000000
// CHECK-EMPTY:
// CHECK-NEXT: Static symbols
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: Exports
// CHECK-EMPTY:
// CHECK-NEXT:  ordinal    name
// CHECK-EMPTY:
// CHECK-NEXT:        1    arm64ec_data_sym

#--- arm64ec-data-sym.s
        .text
        .globl _DllMainCRTStartup
_DllMainCRTStartup:
        ret

        .data
        .globl arm64ec_data_sym
        .p2align 2, 0x0
arm64ec_data_sym:
        .word 0x02020202

        .section .drectve
        .ascii "-export:arm64ec_data_sym,DATA"

#--- arm64-data-sym.s
        .text
        .globl _DllMainCRTStartup
_DllMainCRTStartup:
        ret

        .data
        .globl arm64_data_sym
        .p2align 2, 0x0
arm64_data_sym:
        .word 0x01010101

        .section .drectve
        .ascii "-export:arm64_data_sym,DATA"
