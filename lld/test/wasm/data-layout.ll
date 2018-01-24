; RUN: llc -filetype=obj -mtriple=wasm32-unknown-unknown-wasm %p/Inputs/hello.ll -o %t.hello.o
; RUN: llc -filetype=obj -mtriple=wasm32-unknown-unknown-wasm %s -o %t.o

@foo = hidden global i32 1, align 4
@aligned_bar = hidden global i32 3, align 16

@hello_str = external global i8*
@external_ref = global i8** @hello_str, align 8

; RUN: lld -flavor wasm --allow-undefined -o %t.wasm %t.o %t.hello.o
; RUN: obj2yaml %t.wasm | FileCheck %s

; CHECK:        - Type:            GLOBAL
; CHECK-NEXT:     Globals:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Type:            I32
; CHECK-NEXT:         Mutable:         true
; CHECK-NEXT:         InitExpr:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           66608
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Type:            I32
; CHECK-NEXT:         Mutable:         false
; CHECK-NEXT:         InitExpr:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           66608

; CHECK:         - Type:            DATA
; CHECK-NEXT:     Segments:
; CHECK-NEXT:       - SectionOffset:   7
; CHECK-NEXT:         MemoryIndex:     0
; CHECK-NEXT:         Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           1024
; CHECK-NEXT:         Content:         0100000000000000000000000000000003000000000000001C040000
; CHECK-NEXT:       - SectionOffset:   41
; CHECK-NEXT:         MemoryIndex:     0
; CHECK-NEXT:         Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           1052
; CHECK-NEXT:         Content:         68656C6C6F0A00


; RUN: lld -flavor wasm --relocatable -o %t_reloc.wasm %t.o %t.hello.o
; RUN: obj2yaml %t_reloc.wasm | FileCheck %s -check-prefix=RELOC

; RELOC:        - Type:            GLOBAL
; RELOC-NEXT:     Globals:
; RELOC-NEXT:       - Index:           0
; RELOC-NEXT:         Type:            I32
; RELOC-NEXT:         Mutable:         false
; RELOC-NEXT:         InitExpr:
; RELOC-NEXT:           Opcode:          I32_CONST
; RELOC-NEXT:           Value:           0
; RELOC-NEXT:       - Index:           1
; RELOC-NEXT:         Type:            I32
; RELOC-NEXT:         Mutable:         false
; RELOC-NEXT:         InitExpr:
; RELOC-NEXT:           Opcode:          I32_CONST
; RELOC-NEXT:           Value:           16

; RELOC:       - Type:            DATA
; RELOC-NEXT:     Relocations:
; RELOC-NEXT:       - Type:            R_WEBASSEMBLY_MEMORY_ADDR_I32
; RELOC-NEXT:         Index:           3
; RELOC-NEXT:         Offset:          0x00000018
; RELOC-NEXT:     Segments:
; RELOC-NEXT:       - SectionOffset:   6
; RELOC-NEXT:         MemoryIndex:     0
; RELOC-NEXT:         Offset:
; RELOC-NEXT:           Opcode:          I32_CONST
; RELOC-NEXT:           Value:           0
; RELOC-NEXT:         Content:         '01000000'
; RELOC-NEXT:       - SectionOffset:   15
; RELOC-NEXT:         MemoryIndex:     0
; RELOC-NEXT:         Offset:
; RELOC-NEXT:           Opcode:          I32_CONST
; RELOC-NEXT:           Value:           16
; RELOC-NEXT:         Content:         '03000000'
; RELOC-NEXT:       - SectionOffset:   24
; RELOC-NEXT:         MemoryIndex:     0
; RELOC-NEXT:         Offset:
; RELOC-NEXT:           Opcode:          I32_CONST
; RELOC-NEXT:           Value:           24
; RELOC-NEXT:         Content:         1C000000
; RELOC-NEXT:       - SectionOffset:   33
; RELOC-NEXT:         MemoryIndex:     0
; RELOC-NEXT:         Offset:
; RELOC-NEXT:           Opcode:          I32_CONST
; RELOC-NEXT:           Value:           28
; RELOC-NEXT:         Content:         68656C6C6F0A00

; RELOC:       - Type:            CUSTOM
; RELOC-NEXT:     Name:            linking
; RELOC-NEXT:     DataSize:        35
