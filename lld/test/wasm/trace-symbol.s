# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/ret32.s -o %t.ret32.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.start.o %s
# RUN: wasm-ld -o %t.wasm %t.start.o %t.ret32.o -y ret32 -y _start | FileCheck %s -check-prefix=BOTH
# RUN: wasm-ld -o %t.wasm %t.ret32.o %t.start.o -y ret32 -y _start | FileCheck %s -check-prefix=REVERSED

# check alias
# RUN: wasm-ld -o %t.wasm %t.start.o %t.ret32.o -trace-symbol=_start | FileCheck %s -check-prefixes=JUST-START

.functype ret32 (f32) -> (i32)

.globl  _start
_start:
  .functype _start () -> ()
  f32.const 0.0
  call ret32
  drop
  end_function

# BOTH:          start.o: definition of _start
# BOTH-NEXT:     start.o: reference to ret32
# BOTH-NEXT:     ret32.o: definition of ret32

# REVERSED:      ret32.o: definition of ret32
# REVERSED-NEXT: start.o: definition of _start
# REVERSED-NEXT: start.o: reference to ret32

# JUST-START: start.o: definition of _start
# JUST-START-NOT: ret32
