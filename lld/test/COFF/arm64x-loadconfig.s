// REQUIRES: aarch64
// RUN: split-file %s %t.dir && cd %t.dir

// RUN: llvm-mc -filetype=obj -triple=aarch64-windows test.s -o test.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows loadconfig.s -o loadconfig.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows loadconfig-short.s -o loadconfig-short.obj

// RUN: lld-link -machine:arm64x -out:out.dll -dll -noentry loadconfig.obj test.obj

// RUN: llvm-readobj --coff-load-config out.dll | FileCheck -check-prefix=DYNRELOCS %s
// DYNRELOCS:      DynamicValueRelocTableOffset: 0xC
// DYNRELOCS-NEXT: DynamicValueRelocTableSection: 4
// DYNRELOCS:      DynamicRelocations [
// DYNRELOCS-NEXT:   Version: 0x1
// DYNRELOCS-NEXT:   Arm64X [
// DYNRELOCS-NEXT:     Entry [
// DYNRELOCS-NEXT:       RVA: 0x7C
// DYNRELOCS-NEXT:       Type: VALUE
// DYNRELOCS-NEXT:       Size: 0x2
// DYNRELOCS-NEXT:       Value: 0x8664
// DYNRELOCS-NEXT:     ]
// DYNRELOCS-NEXT:     Entry [
// DYNRELOCS-NEXT:       RVA: 0x150
// DYNRELOCS-NEXT:       Type: VALUE
// DYNRELOCS-NEXT:       Size: 0x4
// DYNRELOCS-NEXT:       Value: 0x0
// DYNRELOCS-NEXT:     ]
// DYNRELOCS-NEXT:     Entry [
// DYNRELOCS-NEXT:       RVA: 0x154
// DYNRELOCS-NEXT:       Type: VALUE
// DYNRELOCS-NEXT:       Size: 0x4
// DYNRELOCS-NEXT:       Value: 0x0
// DYNRELOCS-NEXT:     ]
// DYNRELOCS-NEXT:   ]
// DYNRELOCS-NEXT: ]

// RUN: llvm-readobj --headers out.dll | FileCheck -check-prefix=HEADERS %s
// HEADERS:      BaseRelocationTableRVA: 0x4000
// HEADERS-NEXT: BaseRelocationTableSize: 0xC
// HEADERS:      LoadConfigTableRVA: 0x1000
// HEADERS-NEXT: LoadConfigTableSize: 0x140
// HEADERS:      Name: .reloc (2E 72 65 6C 6F 63 00 00)
// HEADERS-NEXT: VirtualSize: 0x38

// RUN: lld-link -machine:arm64x -out:out-short.dll -dll -noentry loadconfig-short.obj 2>&1 | FileCheck --check-prefix=WARN-RELOC-SIZE %s
// WARN-RELOC-SIZE: lld-link: warning: '_load_config_used' structure too small to include dynamic relocations

#--- test.s
        .data
sym:
        // Emit a basereloc to make the loadconfig test more meaningful.
        .xword sym

#--- loadconfig.s
        .section .rdata,"dr"
        .globl _load_config_used
        .p2align 3, 0
_load_config_used:
        .word 0x140
        .fill 0x13c,1,0

#--- loadconfig-short.s
        .section .rdata,"dr"
        .globl _load_config_used
        .p2align 3, 0
_load_config_used:
        .word 0xe4
        .fill 0xe0,1,0
