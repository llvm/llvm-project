# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld -no-gc-sections --no-entry --export=__data_end %t.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

# Test that the data section is skipped entirely when there are only
# bss segments

  .section  .bss.a,"",@
  .globl  a
a:
  .skip 1000
  .size a, 1000

  .section  .bss.b,"",@
  .globl  b
b:
  .int32  0
  .size b, 4

# CHECK-NOT: - Type:            DATA

#      CHECK:   - Type:            GLOBAL
# CHECK-NEXT:     Globals:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         true
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           67568
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         false
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           2028
# CHECK-NEXT:   - Type:            EXPORT
# CHECK-NEXT:     Exports:
# CHECK-NEXT:       - Name:            memory
# CHECK-NEXT:         Kind:            MEMORY
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:       - Name:            __data_end
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         Index:           1

# CHECK-NOT: DataSegmentNames:
