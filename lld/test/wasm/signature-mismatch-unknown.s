# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/ret32.s -o %t.ret32.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.main.o
# RUN: wasm-ld --fatal-warnings -o %t.wasm %t.ret32.o %t.main.o
# RUN: wasm-ld --fatal-warnings -o %t.wasm %t.main.o %t.ret32.o

# Also test the case where there are two different object files that contains
# references ret32:
# %t.main.o: Does not call ret32 directly; used the wrong signature.
# %t.call-ret32.o: Calls ret32 directly; uses the correct signature.
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/call-ret32.s -o %t.call-ret32.o
# RUN: wasm-ld --export=call_ret32 --fatal-warnings -o %t.wasm %t.main.o %t.call-ret32.o %t.ret32.o
# RUN: wasm-ld --export=call_ret32 --fatal-warnings -o %t.wasm %t.call-ret32.o %t.main.o %t.ret32.o

.functype ret32 () -> ()

.globl _start
_start:
  .functype _start () -> ()
  i32.const 0
  i32.load ptr
  call_indirect () -> (i32)
  drop
  end_function

.section .data.ptr,"",@
.globl ptr
ptr:
  .int32 ret32
  .size ptr, 4
