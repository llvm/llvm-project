// REQUIRES: aarch64

// RUN: llvm-mc -filetype=obj -triple=aarch64-windows %s -o %t.arm64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %s -o %t.arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %S/Inputs/loadconfig-arm64ec.s -o %t-loadconfig-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows %S/Inputs/loadconfig-arm64.s -o %t-loadconfig-arm64.obj

// Check that all symbols are exported in both EC and native views.

// RUN: lld-link -machine:arm64x -lldmingw -dll -noentry -out:%t.dll %t.arm64.obj %t.arm64ec.obj %t-loadconfig-arm64.obj %t-loadconfig-arm64ec.obj

// RUN: llvm-readobj --coff-exports %t.dll | FileCheck --check-prefix=EXP %s
// EXP:      Format: COFF-ARM64X
// EXP-NEXT: Arch: aarch64
// EXP-NEXT: AddressSize: 64bit
// EXP-NEXT: Export {
// EXP-NEXT:   Ordinal: 1
// EXP-NEXT:   Name: sym
// EXP-NEXT:   RVA: 0x2000
// EXP-NEXT: }
// EXP-NEXT: Export {
// EXP-NEXT:   Ordinal: 2
// EXP-NEXT:   Name: sym2
// EXP-NEXT:   RVA: 0x2004
// EXP-NEXT: }
// EXP-NEXT: HybridObject {
// EXP-NEXT:   Format: COFF-ARM64EC
// EXP-NEXT:   Arch: aarch64
// EXP-NEXT:   AddressSize: 64bit
// EXP-NEXT:   Export {
// EXP-NEXT:     Ordinal: 1
// EXP-NEXT:     Name: sym
// EXP-NEXT:     RVA: 0x2008
// EXP-NEXT:   }
// EXP-NEXT:   Export {
// EXP-NEXT:     Ordinal: 2
// EXP-NEXT:     Name: sym2
// EXP-NEXT:     RVA: 0x200C
// EXP-NEXT:   }
// EXP-NEXT: }

// Check that an explicit export in the EC view is respected, preventing symbols from being auto-exported in both EC and native views.

// RUN: lld-link -machine:arm64x -lldmingw -dll -noentry -out:%t2.dll %t.arm64.obj %t.arm64ec.obj -export:sym \
// RUN:          %t-loadconfig-arm64.obj %t-loadconfig-arm64ec.obj

// RUN: llvm-readobj --coff-exports %t2.dll | FileCheck --check-prefix=EXP2 %s
// EXP2:      Format: COFF-ARM64X
// EXP2-NEXT: Arch: aarch64
// EXP2-NEXT: AddressSize: 64bit
// EXP2-NEXT: HybridObject {
// EXP2-NEXT:   Format: COFF-ARM64EC
// EXP2-NEXT:   Arch: aarch64
// EXP2-NEXT:   AddressSize: 64bit
// EXP2-NEXT:   Export {
// EXP2-NEXT:     Ordinal: 1
// EXP2-NEXT:     Name: sym
// EXP2-NEXT:     RVA: 0x2008
// EXP2-NEXT:   }
// EXP2-NEXT: }

        .data
        .globl sym
sym:
        .word 0
        .globl sym2
sym2:
        .word 0
