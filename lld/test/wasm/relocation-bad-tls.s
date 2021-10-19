# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: not wasm-ld --shared-memory %t.o -o %t.wasm 2>&1 | FileCheck %s

.globl _start
_start:
  .functype _start () -> ()
  i32.const foo@TLSREL
  i32.const bar@TLSREL
  drop
  drop
  end_function

.section  .data,"",@
.globl  foo
foo:
  .int32  0
  .size foo, 4

.section  .bss,"",@
.globl  bar
bar:
  .int32  0
  .size bar, 4

# CHECK: relocation R_WASM_MEMORY_ADDR_TLS_SLEB cannot be used against `foo` in non-TLS section: .data
# CHECK: relocation R_WASM_MEMORY_ADDR_TLS_SLEB cannot be used against `bar` in non-TLS section: .bss
