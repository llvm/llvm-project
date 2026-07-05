# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --export=get_handle %t.o -o %t.wasm

  .globl  get_handle
get_handle:
  .functype get_handle () -> (i32)
  i32.const __dso_handle
  end_function

  .globl  _start
_start:
  .functype _start () -> ()
  end_function
