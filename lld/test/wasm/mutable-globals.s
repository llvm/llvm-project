# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: not wasm-ld %t.o -o %t.wasm 2>&1 | FileCheck %s
# RUN: wasm-ld --features=mutable-globals %t.o -o %t.wasm

.globl _start
_start:
  .functype _start () -> ()
  i32.const 1
  global.set foo
  end_function

.globaltype foo, i32
.import_module foo, env
.import_name foo, foo

# CHECK: error: mutable global imported but 'mutable-globals' feature not present in inputs: `foo`. Use --no-check-features to suppress.
