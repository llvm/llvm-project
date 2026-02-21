# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld -no-gc-sections -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.wasm | FileCheck %s --check-prefix=DIS

.globl _start
_start:
  .functype _start () -> (i32)
  global.get tls_sym@GOT@TLS
  end_function

.section  .tdata.tls_sym,"",@
.globl  tls_sym
tls_sym:
  .int32  1
  .size tls_sym, 4

.section  .custom_section.target_features,"",@
  .int8 2
  .int8 43
  .int8 30
  .ascii  "component-model-thread-context"
  .int8 43
  .int8 11
  .ascii  "bulk-memory"


# CHECK: Name: __init_tls_base
# DIS: __wasm_init_tls