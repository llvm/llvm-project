# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/ret32.s -o %t.ret32.o
# RUN: wasm-ld --emit-relocs -o %t.wasm %t.o %t.ret32.o
# RUN: obj2yaml %t.wasm | FileCheck %s

.functype ret32 (f32) -> (i32)

unused_function:
  .functype unused_function () -> ()
  end_function

.globl _start
_start:
  .functype _start () -> ()
  f32.const 0.0
  call ret32
  drop
  i32.const foo
  drop
  i32.const __stack_low
  drop
  end_function

.section .bss.data,"",@
.p2align        2
foo:
  .int32  0
  .size   foo, 4

.section .debug_info,"",@
.p2align 2
.int32 unused_function
.int32 _start
.int32 0

# CHECK:        - Type:            CODE
# CHECK-NEXT:     Relocations:
# CHECK-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
# CHECK-NEXT:         Index:           1
# CHECK-NEXT:         Offset:          0x9

# CHECK:        - Type:            DATA
# CHECK-NEXT:     Segments:
# CHECK-NEXT:       - SectionOffset:   8
# CHECK-NEXT:         InitFlags:       0
# CHECK-NEXT:         Offset:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           65536
# CHECK-NEXT:         Content:         '00000000'

# There should be a single relocation in this section (just the live symbol)
# CHECK-NEXT:  - Type:            CUSTOM
# CHECK-NEXT:    Relocations:
# CHECK-NEXT:      - Type:            R_WASM_FUNCTION_OFFSET_I32
# CHECK-NEXT:        Index:           0
# CHECK-NEXT:        Offset:          0x4
# CHECK-NEXT:    Name:            .debug_info
# CHECK-NEXT:    Payload:         FFFFFFFF0200000000000000

# CHECK:        - Type:            CUSTOM
# CHECK-NEXT:     Name:            linking
# CHECK-NEXT:     Version:         2
# CHECK-NEXT:     SymbolTable:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Name:            _start
# CHECK-NEXT:         Flags:           [  ]
# CHECK-NEXT:         Function:        0
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Name:            ret32
# CHECK-NEXT:         Flags:           [  ]
# CHECK-NEXT:         Function:        1
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Kind:            DATA
# CHECK-NEXT:         Name:            __stack_low
# CHECK-NEXT:         Flags:           [ VISIBILITY_HIDDEN, ABSOLUTE ]
# CHECK-NEXT:         Size:            0
# CHECK-NEXT:       - Index:           3
