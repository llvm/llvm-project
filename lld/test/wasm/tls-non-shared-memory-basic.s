# Simplified version of tls-non-shared-memory.s that does not reference
# __tls_base

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s

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

# RUN: wasm-ld --experimental-pic -shared -o %t.so %t.o
# RUN: obj2yaml %t.so | FileCheck %s --check-prefix=PIC

#      CHECK:  - Type:            DATA
# CHECK-NEXT:    Segments:
# CHECK-NEXT:      - SectionOffset:   7
# CHECK-NEXT:        InitFlags:       0
# CHECK-NEXT:        Offset:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           1024
# CHECK-NEXT:        Content:         2B000000
# CHECK-NEXT:  - Type:            CUSTOM
# CHECK-NOT:   - Type:            IMPORT


# In PIC mode we expect TLS data and non-TLS data to be merged into
# a single segment which is initialized via the  __memory_base import

#      PIC:  - Type:            IMPORT
# PIC-NEXT:    Imports:
# PIC-NEXT:      - Module:          env
# PIC-NEXT:        Field:           memory
# PIC-NEXT:        Kind:            MEMORY
# PIC-NEXT:        Memory:
# PIC-NEXT:          Minimum:         0x1
# PIC-NEXT:      - Module:          env
# PIC-NEXT:        Field:           __memory_base
# PIC-NEXT:        Kind:            GLOBAL
# PIC-NEXT:        GlobalType:      I32

#      PIC:  - Type:            DATA
# PIC-NEXT:    Segments:
# PIC-NEXT:      - SectionOffset:   6
# PIC-NEXT:        InitFlags:       0
# PIC-NEXT:        Offset:
# PIC-NEXT:          Opcode:          GLOBAL_GET
# PIC-NEXT:          Index:           0
# PIC-NEXT:        Content:         2B000000
# PIC-NEXT:  - Type:            CUSTOM
