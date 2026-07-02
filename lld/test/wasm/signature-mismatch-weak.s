# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/weak-symbol1.s -o %t.weak.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/strong-symbol.s -o %t.strong.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: wasm-ld -o %t.wasm %t.o %t.strong.o %t.weak.o 2>&1 | FileCheck %s

.functype weakFn () -> (i32)
.globl _start
_start:
  .functype _start () -> ()
  call weakFn
  drop
  end_function

# CHECK: warning: function signature mismatch: weakFn
# CHECK-NEXT: >>> defined as () -> i32 in {{.*}}signature-mismatch-weak.s.tmp.o
# CHECK-NEXT: >>> defined as () -> i64 in {{.*}}signature-mismatch-weak.s.tmp.strong.o
