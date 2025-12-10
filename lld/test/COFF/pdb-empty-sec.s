// REQUIRES: x86

// RUN: llvm-mc -filetype=obj -triple=x86_64-windows %s -o %t.obj
// RUN: lld-link -dll -noentry -debug %t.obj -out:%t.dll
// RUN: llvm-pdbutil dump -publics %t.pdb | FileCheck %s

// CHECK:       Records
// CHECK-NEXT:       0 | S_PUB32 [size = 20] `func`
// CHECK-NEXT:           flags = none, addr = 0001:0000
// CHECK-NEXT:      20 | S_PUB32 [size = 20] `sym`
// CHECK-NEXT:           flags = none, addr = 0000:0000

        .globl sym
        .data
sym:
        .text
        .globl func
func:
        ret
