# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: not wasm-ld -shared --experimental-pic --shared-memory -o %t.wasm %t.o 2>&1 | FileCheck %s

# CHECK: error: {{.*}}.o: TLS symbol is undefined, but TLS symbols cannot yet be imported: `foo`

.globl _start
_start:
  .functype _start () -> ()
  i32.const foo@TLSREL
  drop
  end_function

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
