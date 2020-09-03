# Verify that the entry point signature can be flexible.
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld -o %t1.wasm %t.o

  .globl  _start
_start:
  .functype _start (i64) -> (f32)
  end_function
