; RUN: llc -filetype=obj %s -o %t.o
; RUN: wasm-ld -no-gc-sections --no-entry -o %t.wasm %t.o
; RUN: obj2yaml %t.wasm | FileCheck %s
; RUN: wasm-ld -no-gc-sections --no-entry -o %t_reloc.o %t.o --relocatable
; RUN: obj2yaml %t_reloc.o | FileCheck -check-prefix RELOC %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

@bss = hidden global i32 zeroinitializer, align 4
@foo = hidden global i32 zeroinitializer, section "WowZero!", align 4
@bar = hidden constant i32 42, section "MyAwesomeSection", align 4
@baz = hidden global i32 7, section "AnotherGreatSection", align 4

; CHECK-LABEL: - Type:            DATA
; CHECK-NEXT:    Segments:
; CHECK-NEXT:      - SectionOffset:   7
; CHECK-NEXT:        InitFlags:       0
; CHECK-NEXT:        Offset:
; CHECK-NEXT:          Opcode:          I32_CONST
; CHECK-NEXT:          Value:           1024
; CHECK-NEXT:        Content:         '00000000'
; CHECK-NEXT:      - SectionOffset:   17
; CHECK-NEXT:        InitFlags:       0
; CHECK-NEXT:        Offset:
; CHECK-NEXT:          Opcode:          I32_CONST
; CHECK-NEXT:          Value:           1028
; CHECK-NEXT:        Content:         2A000000
; CHECK-NEXT:      - SectionOffset:   27
; CHECK-NEXT:        InitFlags:       0
; CHECK-NEXT:        Offset:
; CHECK-NEXT:          Opcode:          I32_CONST
; CHECK-NEXT:          Value:           1032
; CHECK-NEXT:        Content:         '07000000'
; CHECK-NEXT:      - SectionOffset:   37
; CHECK-NEXT:        InitFlags:       0
; CHECK-NEXT:        Offset:
; CHECK-NEXT:          Opcode:          I32_CONST
; CHECK-NEXT:          Value:           1036
; CHECK-NEXT:        Content:         '00000000'

; RELOC-LABEL: SegmentInfo:
; RELOC-NEXT:    - Index:           0
; RELOC-NEXT:      Name:            'WowZero!'
; RELOC-NEXT:      Alignment:       2
; RELOC-NEXT:      Flags:           [  ]
; RELOC-NEXT:    - Index:           1
; RELOC-NEXT:      Name:            MyAwesomeSection
; RELOC-NEXT:      Alignment:       2
; RELOC-NEXT:      Flags:           [  ]
; RELOC-NEXT:    - Index:           2
; RELOC-NEXT:      Name:            AnotherGreatSection
; RELOC-NEXT:      Alignment:       2
; RELOC-NEXT:      Flags:           [  ]
; RELOC-NEXT:    - Index:           3
; RELOC-NEXT:      Name:            .bss.bss
; RELOC-NEXT:      Alignment:       2
; RELOC-NEXT:      Flags:           [  ]
