# RUN: llvm-mc -filetype=obj -triple=wasm32 %s -o %t.o

## A positive integer is allowed.
# RUN: wasm-ld --no-entry %t.o -o /dev/null
# RUN: wasm-ld --no-entry --threads=1 %t.o -o /dev/null
# RUN: wasm-ld --no-entry --threads=2 %t.o -o /dev/null

# RUN: not wasm-ld --threads=all %t.o -o /dev/null 2>&1 | FileCheck %s -DN=all
# RUN: not wasm-ld --threads=0 %t.o -o /dev/null 2>&1 | FileCheck %s -DN=0
# RUN: not wasm-ld --threads=-1 %t.o -o /dev/null 2>&1 | FileCheck %s -DN=-1

# CHECK: error: --threads: expected a positive integer, but got '[[N]]'
