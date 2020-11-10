# Test that linking without shared memory causes __tls_base to be
# interlized

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s

.globaltype __tls_base, i32

.globl get_tls1
get_tls1:
  .functype get_tls1 () -> (i32)
  global.get __tls_base
  i32.const tls1
  i32.add
  end_function

.section  .data.no_tls,"",@
.globl  no_tls
.p2align  2
no_tls:
  .int32  42
  .size no_tls, 4

.section  .tdata.tls1,"",@
.globl  tls1
.p2align  2
tls1:
  .int32  43
  .size tls1, 2

.section  .custom_section.target_features,"",@
  .int8 2
  .int8 43
  .int8 7
  .ascii  "atomics"
  .int8 43
  .int8 11
  .ascii  "bulk-memory"

# RUN: wasm-ld --no-gc-sections --no-entry -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s

#      CHECK:   - Type:            GLOBAL
# __stack_pointer
# CHECK-NEXT:     Globals:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         true
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           66576
# __tls_base
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         false
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           1028
# CHECK-NEXT:   - Type:            EXPORT

#      CHECK:  - Type:            DATA
# .data
# CHECK-NEXT:    Segments:
# CHECK-NEXT:      - SectionOffset:   7
# CHECK-NEXT:        InitFlags:       0
# CHECK-NEXT:        Offset:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           1024
# CHECK-NEXT:        Content:         2A000000
# .tdata
# CHECK-NEXT:      - SectionOffset:   17
# CHECK-NEXT:        InitFlags:       0
# CHECK-NEXT:        Offset:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           1028
# CHECK-NEXT:        Content:         2B000000
