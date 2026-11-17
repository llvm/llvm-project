// RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
// RUN: wasm-ld -shared -o %t.so %t.o

// RUN: wasm-ld -pie -o /dev/null %t.o %t.so
// RUN: not wasm-ld -o /dev/null -static %t.o %t.so 2>&1 | FileCheck %s

// CHECK: attempted static link of dynamic object

.global _start
_start:
  .functype _start () -> ()
  end_function
