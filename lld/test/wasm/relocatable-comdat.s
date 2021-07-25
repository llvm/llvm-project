# RUN: llvm-mc -triple=wasm32 -filetype=obj %p/Inputs/comdat-data.s -o %t1.o
# RUN: llvm-mc -triple=wasm32 -filetype=obj %s -o %t.o
# RUN: wasm-ld --relocatable -o %t.wasm %t.o %t1.o
# RUN: obj2yaml %t.wasm | FileCheck %s


        .globl  _start
        .type  _start,@function
_start:
        .functype _start () -> ()
        i32.const 0
        i32.load foo
        drop
        end_function


.section  .data.foo,"",@
foo:
        .int32 42
        .size foo, 4

# Verify that .data.foo in this file is not merged with comdat .data.foo
# section in Inputs/comdat-data.s.

#      CHECK:   - Type:            DATA
# CHECK-NEXT:     Segments:
# CHECK-NEXT:       - SectionOffset:   6
# CHECK-NEXT:         InitFlags:       0
# CHECK-NEXT:         Offset:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           0
# CHECK-NEXT:         Content:         2A000000
# CHECK-NEXT:       - SectionOffset:   15
# CHECK-NEXT:         InitFlags:       0
# CHECK-NEXT:         Offset:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           4
# CHECK-NEXT:         Content:         2A0000002B000000

#      CHECK:    SegmentInfo:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Name:            .data.foo
# CHECK-NEXT:        Alignment:       0
# CHECK-NEXT:        Flags:           [  ]
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        Name:            .data.foo
# CHECK-NEXT:        Alignment:       0
# CHECK-NEXT:        Flags:           [  ]
