# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s

.globaltype __tls_base, i32
.globaltype __tls_align, i32, immutable

.globl tls1_addr
tls1_addr:
  .functype tls1_addr () -> (i32)
  global.get __tls_base
  i32.const tls1
  i32.add
  end_function

.globl tls2_addr
tls2_addr:
  .functype tls2_addr () -> (i32)
  global.get __tls_base
  i32.const tls2
  i32.add
  end_function

.globl tls_align
tls_align:
  .functype tls_align () -> (i32)
  global.get __tls_align
  end_function

.section  .bss.no_tls,"",@
.globl  no_tls
.p2align  2
no_tls:
  .int32  0
  .size no_tls, 4

.section  .tdata.tls1,"",@
.globl  tls1
.p2align  2
tls1:
  .int32  1
  .size tls1, 4

.section  .tdata.tls2,"",@
.globl  tls2
.p2align  4
tls2:
  .int32  1
  .size tls2, 4

.section  .custom_section.target_features,"",@
  .int8 2
  .int8 43
  .int8 7
  .ascii  "atomics"
  .int8 43
  .int8 11
  .ascii  "bulk-memory"

# RUN: wasm-ld -no-gc-sections --shared-memory --max-memory=131072 --no-entry -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s

# CHECK:      - Type:            GLOBAL
# CHECK-NEXT:   Globals:
# CHECK-NEXT:     - Index:           0
# CHECK-NEXT:       Type:            I32
# CHECK-NEXT:       Mutable:         true
# CHECK-NEXT:       InitExpr:
# CHECK-NEXT:         Opcode:          I32_CONST
# CHECK-NEXT:         Value:           66592

# __tls_base
# CHECK-NEXT:     - Index:           1
# CHECK-NEXT:       Type:            I32
# CHECK-NEXT:       Mutable:         true
# CHECK-NEXT:       InitExpr:
# CHECK-NEXT:         Opcode:          I32_CONST
# CHECK-NEXT:         Value:           0

# __tls_size
# CHECK-NEXT:     - Index:           2
# CHECK-NEXT:       Type:            I32
# CHECK-NEXT:       Mutable:         false
# CHECK-NEXT:       InitExpr:
# CHECK-NEXT:         Opcode:          I32_CONST
# CHECK-NEXT:         Value:           20

# __tls_align
# CHECK-NEXT:     - Index:           3
# CHECK-NEXT:       Type:            I32
# CHECK-NEXT:       Mutable:         false
# CHECK-NEXT:       InitExpr:
# CHECK-NEXT:         Opcode:          I32_CONST
# CHECK-NEXT:         Value:           16
