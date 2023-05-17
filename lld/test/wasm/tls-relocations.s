# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s

.section data,"",@
  .int32 41
data_sym:
  .int32 42
  .size data_sym, 4

# TLS data section of size 16 with as relocations at offset 8 and 12
.section tls_sec,"T",@
.globl  tls_sym
.p2align  2
  .int32 0x50
tls_sym:
  .int32 0x51
  .int32 data_sym
  .int32 tls_sym
  .size tls_sym, 4

.section  .custom_section.target_features,"",@
  .int8 2
  .int8 43
  .int8 7
  .ascii  "atomics"
  .int8 43
  .int8 11
  .ascii  "bulk-memory"

# RUN: wasm-ld --experimental-pic -pie -no-gc-sections --shared-memory --no-entry -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn --no-leading-addr %t.wasm | FileCheck --check-prefix=ASM %s --

# CHECK:       - Type:            GLOBAL

# __tls_base
# CHECK-NEXT:    Globals:
# CHECK-NEXT:      - Index:           3
# CHECK-NEXT:        Type:            I32
# CHECK-NEXT:        Mutable:         true
# CHECK-NEXT:        InitExpr:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           0

# __tls_size
# CHECK-NEXT:      - Index:           4
# CHECK-NEXT:        Type:            I32
# CHECK-NEXT:        Mutable:         false
# CHECK-NEXT:        InitExpr:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           16

# __tls_align
# CHECK-NEXT:      - Index:           5
# CHECK-NEXT:        Type:            I32
# CHECK-NEXT:        Mutable:         false
# CHECK-NEXT:        InitExpr:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           4

# ASM:       <__wasm_init_tls>:
# ASM-EMPTY:
# ASM-NEXT:                 local.get 0
# ASM-NEXT:                 global.set  3
# ASM-NEXT:                 local.get 0
# ASM-NEXT:                 i32.const 0
# ASM-NEXT:                 i32.const 16
# ASM-NEXT:                 memory.init 0, 0
# call to __wasm_apply_tls_relocs
# ASM-NEXT:                 call  4
# ASM-NEXT:                 end

# ASM: <__wasm_apply_tls_relocs>:
# ASM-EMPTY:
# ASM-NEXT:                 i32.const 8
# ASM-NEXT:                 global.get  3
# ASM-NEXT:                 i32.add
# ASM-NEXT:                 global.get  1
# ASM-NEXT:                 i32.const 20
# ASM-NEXT:                 i32.add
# ASM-NEXT:                 i32.store 0
# ASM-NEXT:                 i32.const 12
# ASM-NEXT:                 global.get  3
# ASM-NEXT:                 i32.add
# ASM-NEXT:                 global.get  3
# ASM-NEXT:                 i32.const 4
# ASM-NEXT:                 i32.add
# ASM-NEXT:                 i32.store 0
# ASM-NEXT:                 end
