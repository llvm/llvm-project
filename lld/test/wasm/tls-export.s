# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld -no-gc-sections --shared-memory --no-entry -o %t.wasm %t.o
# RUN: not wasm-ld --shared-memory --no-entry --export=tls1 -o %t.wasm %t.o 2>&1 | FileCheck %s
# With --export-all we ignore TLS symbols so we don't expect an error here
# RUN: wasm-ld --shared-memory --no-entry --export-all -o %t.wasm %t.o

# CHECK: error: TLS symbols cannot yet be exported: `tls1`

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
