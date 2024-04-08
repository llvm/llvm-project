# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t.64.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 %p/Inputs/riscv-split-stack-callee.s -o %t.64.2.o
# RUN: not ld.lld %t.64.o %t.64.2.o -o %t.64 2>&1 | FileCheck %s
# CHECK: ld.lld: error: Target doesn't support split stacks.

        .globl  _start
        .type   _start,@function
_start:
        call    test
	ret
end:
        .size   _start, end-_start
        .section        ".note.GNU-split-stack","",@progbits
