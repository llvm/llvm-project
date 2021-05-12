// RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
// RUN: wasm-ld -O1 %t.o -o %t.wasm --no-gc-sections --no-entry
// RUN: obj2yaml %t.wasm | FileCheck %s --check-prefixes=COMMON,MERGE

// Check that the default is the same as -O1 (since we default to -O1)
// RUN: wasm-ld %t.o -o %t.wasm --no-gc-sections --no-entry
// RUN: obj2yaml %t.wasm | FileCheck %s --check-prefixes=COMMON,MERGE

// Check that -O0 disables merging
// RUN: wasm-ld -O0 %t.o -o %t2.wasm --no-gc-sections --no-entry
// RUN: obj2yaml %t2.wasm | FileCheck --check-prefixes=COMMON,NOMERGE %s

        .section .rodata1,"S",@
        .asciz "abc"
foo:
        .ascii "a"
        .size foo, 1
bar:
        .asciz "bc"
        .asciz "bc"
        .size bar, 4

        .section .rodata_relocs,"",@
negative_addend:
        .int32 foo-10
        .size negative_addend, 4

.globl foo
.globl bar
.export_name    foo, foo
.export_name    bar, bar

//       COMMON:  - Type:            GLOBAL
//  COMMON-NEXT:    Globals:
//  COMMON-NEXT:      - Index:           0
//  COMMON-NEXT:        Type:            I32
//  COMMON-NEXT:        Mutable:         true
//  COMMON-NEXT:        InitExpr:
//  COMMON-NEXT:          Opcode:          I32_CONST
//  COMMON-NEXT:          Value:           66576
//  COMMON-NEXT:      - Index:           1
//  COMMON-NEXT:        Type:            I32
//  COMMON-NEXT:        Mutable:         false
//  COMMON-NEXT:        InitExpr:
//  COMMON-NEXT:          Opcode:          I32_CONST
//   MERGE-NEXT:          Value:           1024
// NOMERGE-NEXT:          Value:           1028
//  COMMON-NEXT:      - Index:           2
//  COMMON-NEXT:        Type:            I32
//  COMMON-NEXT:        Mutable:         false
//  COMMON-NEXT:        InitExpr:
//  COMMON-NEXT:          Opcode:          I32_CONST
//   MERGE-NEXT:          Value:           1025
// NOMERGE-NEXT:          Value:           1029
//  COMMON-NEXT:  - Type:            EXPORT
//  COMMON-NEXT:    Exports:
//  COMMON-NEXT:      - Name:            memory
//  COMMON-NEXT:        Kind:            MEMORY
//  COMMON-NEXT:        Index:           0
//  COMMON-NEXT:      - Name:            foo
//  COMMON-NEXT:        Kind:            GLOBAL
//  COMMON-NEXT:        Index:           1
//  COMMON-NEXT:      - Name:            bar
//  COMMON-NEXT:        Kind:            GLOBAL
//  COMMON-NEXT:        Index:           2

//
//       COMMON:  - Type:            DATA
//  COMMON-NEXT:    Segments:
//  COMMON-NEXT:      - SectionOffset:   7
//  COMMON-NEXT:        InitFlags:       0
//  COMMON-NEXT:        Offset:
//  COMMON-NEXT:          Opcode:          I32_CONST
//  COMMON-NEXT:          Value:           1024
//   MERGE-NEXT:          Content:         '61626300'
// NOMERGE-NEXT:          Content:         '6162630061626300626300'
