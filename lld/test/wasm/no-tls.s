# Testing that __tls_size and __tls_align are correctly emitted when there are
# no thread_local variables.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s

.globl  _start
_start:
  .functype _start () -> ()
  end_function

.section  .custom_section.target_features,"",@
  .int8 2
  .int8 43
  .int8 7
  .ascii  "atomics"
  .int8 43
  .int8 11
  .ascii  "bulk-memory"

# RUN: wasm-ld -no-gc-sections --shared-memory --max-memory=131072 --allow-undefined -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s
# CHECK:       - Type:            GLOBAL
# CHECK-NEXT:    Globals:

# __stack_pointer
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Type:            I32
# CHECK-NEXT:        Mutable:         true
# CHECK-NEXT:        InitExpr:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           66576

# __tls_base
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        Type:            I32
# CHECK-NEXT:        Mutable:         true
# CHECK-NEXT:        InitExpr:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           0

# __tls_size
# CHECK-NEXT:      - Index:           2
# CHECK-NEXT:        Type:            I32
# CHECK-NEXT:        Mutable:         false
# CHECK-NEXT:        InitExpr:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           0

# __tls_align
# CHECK-NEXT:      - Index:           3
# CHECK-NEXT:        Type:            I32
# CHECK-NEXT:        Mutable:         false
# CHECK-NEXT:        InitExpr:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           0
