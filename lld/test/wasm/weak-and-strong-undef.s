# Test that when a symbol (foo) is both weakly and strongly referenced
# the strong undefined symbol always generates an error, whichever object
# file is seen first.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %S/Inputs/weak-undefined.s -o %t2.o
# RUN: not wasm-ld %t1.o %t2.o -o /dev/null 2>&1 | FileCheck %s
# RUN: not wasm-ld %t2.o %t1.o -o /dev/null 2>&1 | FileCheck %s

# CHECK: undefined symbol: foo

_start:
    .globl _start
    .functype _start () -> ()
    call foo
    end_function

.functype foo () -> ()
