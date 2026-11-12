# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/ret32.s -o %t.ret32.o
# RUN: wasm-ld -o %t.wasm %t.o %t.ret32.o 2>&1 | FileCheck %s -check-prefix=CHECK-WARN
# RUN: not wasm-ld --fatal-warnings -o %t.wasm %t.o %t.ret32.o 2>&1 | FileCheck %s -check-prefix=CHECK-FATAL

# CHECK-WARN: warning: function signature mismatch: ret32
# CHECK-FATAL: error: function signature mismatch: ret32

.functype ret32 (f32, i64, i32) -> (i32)

.globl  _start
_start:
  .functype _start () -> ()
  f32.const 1.0
  i64.const 2
  i32.const 3
  call ret32
  drop
  end_function
