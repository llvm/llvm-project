# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld -shared --experimental-pic -o %t.so %t.o
# RUN: obj2yaml %t.so | FileCheck %s

.section  .tdata.tls1,"",@
.globl  tls1
.p2align  2
tls1:
  .int32  1
  .size tls1, 4

.section  .custom_section.target_features,"",@
  .int8 3
  .int8 43
  .int8 7
  .ascii  "atomics"
  .int8 43
  .int8 11
  .ascii  "bulk-memory"
  .int8 43
  .int8 15
  .ascii "mutable-globals"

#      CHECK:    ExportInfo:
# CHECK-NEXT:      - Name:            tls1
# CHECK-NEXT:        Flags:           [ TLS ]
# CHECK-NEXT:  - Type:            TYPE

#      CHECK:  - Type:            GLOBAL
# CHECK-NEXT:    Globals:
# CHECK-NEXT:      - Index:           2
# CHECK-NEXT:        Type:            I32
# CHECK-NEXT:        Mutable:         false
# CHECK-NEXT:        InitExpr:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           0

#      CHECK:  - Type:            EXPORT
# CHECK-NEXT:    Exports:
# CHECK-NEXT:      - Name:            __wasm_call_ctors
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           0
# CHECK-NEXT:      - Name:            __wasm_apply_data_relocs
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           1
# CHECK-NEXT:      - Name:            tls1
# CHECK-NEXT:        Kind:            GLOBAL
# CHECK-NEXT:        Index:           2
