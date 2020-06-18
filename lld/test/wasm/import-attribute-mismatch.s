# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t1.o %s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %S/Inputs/import-attributes.s -o %t2.o
# RUN: not wasm-ld --export call_foo --allow-undefined -o %t.wasm %t1.o %t2.o 2>&1 | FileCheck %s

  .globl  _start
_start:
  .functype _start () -> ()
  call  foo
  end_function

.functype foo () -> ()
.import_module  foo, bar

# CHECK: wasm-ld: error: import module mismatch for symbol: foo
# CHECK: >>> defined as bar in {{.*}}1.o
# CHECK: >>> defined as baz in {{.*}}2.o
