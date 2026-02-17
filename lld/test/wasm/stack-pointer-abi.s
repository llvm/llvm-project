# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --component-model-thread-context -o %t-component-model.wasm %t.o
# RUN: obj2yaml %t-component-model.wasm | FileCheck %s --check-prefix=WITH
# RUN: wasm-ld -o %t-original.wasm %t.o
# RUN: obj2yaml %t-original.wasm | FileCheck %s --check-prefix=WITHOUT

.globl _start
_start:
  .functype _start () -> ()
  end_function

# WITH: Name: __init_stack_pointer
# WITHOUT: Name: __stack_pointer