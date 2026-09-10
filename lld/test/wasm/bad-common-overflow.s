# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: not wasm-ld --no-entry --export=foo --export=bar %t.o -o /dev/null 2>&1 | FileCheck %s

# CHECK: error: common symbols section size overflow

# Each common symbol is within the 4G limit but the combination overflow.
.comm foo, 4294967295, 2
.comm bar, 16, 2
