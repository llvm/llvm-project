# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/ret32.s -o %t.ret32.o
# RUN: not wasm-ld -o %t.wasm %t.o %t.ret32.o 2>&1 | FileCheck %s

.weak ret32

# CHECK: error: symbol type mismatch: ret32
# CHECK: >>> defined as WASM_SYMBOL_TYPE_DATA in {{.*}}symbol-type-mismatch.s.tmp.o
# CHECK: >>> defined as WASM_SYMBOL_TYPE_FUNCTION in {{.*}}.ret32.o
