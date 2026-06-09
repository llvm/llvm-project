# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --libcall-thread-context --no-gc-sections -o %t.libcall.wasm %t.o
# RUN: obj2yaml %t.libcall.wasm | FileCheck %s --check-prefix=LIBCALL
# RUN: wasm-ld --no-gc-sections -o %t.global.wasm %t.o
# RUN: obj2yaml %t.global.wasm | FileCheck %s --check-prefix=GLOBAL

  .globl  _start
_start:
  .functype _start () -> ()
  end_function

# LIBCALL: Name: __init_stack_pointer
# GLOBAL: Name: __stack_pointer
