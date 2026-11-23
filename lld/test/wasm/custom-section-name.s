# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -mattr=-bulk-memory %s -o %t.o
# RUN: wasm-ld -no-gc-sections --no-entry -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s --check-prefixes=CHECK,NO-BSS
# RUN: wasm-ld -no-gc-sections --no-entry --import-memory -o %t.bss.wasm %t.o
# RUN: obj2yaml %t.bss.wasm | FileCheck %s --check-prefixes=CHECK,BSS
# RUN: wasm-ld -no-gc-sections --no-entry -o %t_reloc.o %t.o --relocatable
# RUN: obj2yaml %t_reloc.o | FileCheck -check-prefix RELOC %s

.section .bss.bss,"",@
.globl  bss
.p2align  2, 0x0
bss:
  .int32  0
  .size bss, 4

.section "WowZero!","",@
.globl  foo
.p2align 2
foo:
  .int32  0
  .size foo, 4

.section MyAwesomeSection,"",@
.globl  bar
.p2align 2
bar:
  .int32  42
  .size bar, 4

.section AnotherGreatSection,"",@
.globl  baz
.p2align 2
baz:
  .int32  7
  .size baz, 4

# CHECK-LABEL: - Type:            DATA
# CHECK-NEXT:    Segments:
# CHECK-NEXT:      - SectionOffset:   8
# CHECK-NEXT:        InitFlags:       0
# CHECK-NEXT:        Offset:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           65536
# CHECK-NEXT:        Content:         '00000000'
# CHECK-NEXT:      - SectionOffset:   19
# CHECK-NEXT:        InitFlags:       0
# CHECK-NEXT:        Offset:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           65540
# CHECK-NEXT:        Content:         2A000000
# CHECK-NEXT:      - SectionOffset:   30
# CHECK-NEXT:        InitFlags:       0
# CHECK-NEXT:        Offset:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           65544
# CHECK-NEXT:        Content:         '07000000'
# BSS-NEXT:        - SectionOffset:   41
# BSS-NEXT:          InitFlags:       0
# BSS-NEXT:          Offset:
# BSS-NEXT:            Opcode:          I32_CONST
# BSS-NEXT:            Value:           65548
# BSS-NEXT:          Content:         '00000000'
# NO-BSS-NOT:      - SectionOffset:

# RELOC-LABEL: SegmentInfo:
# RELOC-NEXT:    - Index:           0
# RELOC-NEXT:      Name:            'WowZero!'
# RELOC-NEXT:      Alignment:       2
# RELOC-NEXT:      Flags:           [  ]
# RELOC-NEXT:    - Index:           1
# RELOC-NEXT:      Name:            MyAwesomeSection
# RELOC-NEXT:      Alignment:       2
# RELOC-NEXT:      Flags:           [  ]
# RELOC-NEXT:    - Index:           2
# RELOC-NEXT:      Name:            AnotherGreatSection
# RELOC-NEXT:      Alignment:       2
# RELOC-NEXT:      Flags:           [  ]
# RELOC-NEXT:    - Index:           3
# RELOC-NEXT:      Name:            .bss.bss
# RELOC-NEXT:      Alignment:       2
# RELOC-NEXT:      Flags:           [  ]
