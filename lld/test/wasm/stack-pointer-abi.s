# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-component-model.o %S/Inputs/use-component-model-thread-context.s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-global.o %S/Inputs/disallow-component-model-thread-context.s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --component-model-thread-context -o %t-component-model.wasm %t-component-model.o %t.o
# RUN: obj2yaml %t-component-model.wasm | FileCheck %s --check-prefix=COMPONENT-MODEL
# RUN: wasm-ld -o %t-original.wasm %t-global.o %t.o
# RUN: obj2yaml %t-original.wasm | FileCheck %s --check-prefix=GLOBAL

  .globl  _start
_start:
  .functype _start () -> ()
  end_function

# COMPONENT-MODEL: Name: __init_stack_pointer
# GLOBAL: Name: __stack_pointer