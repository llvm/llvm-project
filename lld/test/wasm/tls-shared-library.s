# Thread-local variables defined in a shared library cannot be referenced from
# another module, since TLS relocations are resolved against the referencing
# module's own `__tls_base`.

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/lib.o %t/lib.s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/main.o %t/main.s
# RUN: wasm-ld -shared --experimental-pic --shared-memory -o %t/lib.so %t/lib.o
# RUN: not wasm-ld -shared --experimental-pic --shared-memory -o /dev/null %t/main.o %t/lib.so 2>&1 | FileCheck %s
# RUN: not wasm-ld -pie --experimental-pic --shared-memory --no-entry --export=get_tls1 -o /dev/null %t/main.o %t/lib.so 2>&1 | FileCheck %s

# Without shared memory TLS relocations are otherwise accepted against any
# symbol, so this used to link silently.
# RUN: not wasm-ld -shared --experimental-pic -o /dev/null %t/main.o %t/lib.so 2>&1 | FileCheck %s

# CHECK: error: {{.*}}main.o: relocation R_WASM_MEMORY_ADDR_TLS_SLEB cannot be used against symbol `tls1` defined in shared library {{.*}}lib.so; thread-local variables cannot be accessed across modules
# CHECK-NOT: error:

#--- lib.s
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

#--- main.s
.globaltype __tls_base, i32

.globl get_tls1
get_tls1:
  .functype get_tls1 () -> (i32)
  global.get __tls_base
  i32.const tls1@TLSREL
  i32.add
  i32.load 0
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
