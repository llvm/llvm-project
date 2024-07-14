# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: not wasm-ld --experimental-pic %t.o -o /dev/null -shared 2>&1 | FileCheck %s

# CHECK: error: {{.*}}: undefined symbol: hidden
.global hidden
.hidden hidden

.global foo
.section .data,"",@
foo:
 .int32 hidden
 .size foo,4
