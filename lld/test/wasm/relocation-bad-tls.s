# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/lib.o %t/lib.s
# RUN: wasm-ld -shared --features=atomics,bulk-memory -o %t/lib.so %t/lib.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/main.o %t/main.s
# RUN: not wasm-ld -pie --shared-memory %t/main.o %t/lib.so -o %t.wasm 2>&1 | FileCheck %s
# RUN: not wasm-ld -pie %t/main.o %t/lib.so -o %t.wasm 2>&1 | FileCheck %s --check-prefix=NO-SHARED-MEM

#--- lib.s
.section  .tdata,"T",@
.globl  qux
qux:
  .int32  0
  .size qux, 4

#--- main.s
.globl _start
_start:
  .functype _start () -> ()
  i32.const foo@TLSREL
  i32.const bar@TLSREL
  i32.const baz@TLSREL
  i32.const qux@TLSREL
  drop
  drop
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
# CHECK: relocation R_WASM_MEMORY_ADDR_TLS_SLEB cannot be used against an undefined symbol `baz`
# CHECK: relocation R_WASM_MEMORY_ADDR_TLS_SLEB cannot be used against a shared library symbol `qux`

# NO-SHARED-MEM-NOT: non-TLS section
# NO-SHARED-MEM: relocation R_WASM_MEMORY_ADDR_TLS_SLEB cannot be used against an undefined symbol `baz`
# NO-SHARED-MEM: relocation R_WASM_MEMORY_ADDR_TLS_SLEB cannot be used against a shared library symbol `qux`
