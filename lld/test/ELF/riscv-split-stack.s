# REQUIRES: riscv

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=riscv64 %t/main.s -o %t.64.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 %t/callee.s -o %t.64.2.o
# RUN: not ld.lld %t.64.o %t.64.2.o -o %t.64 2>&1 | FileCheck %s
# CHECK: error: target doesn't support split stacks

#--- main.s
        .globl  _start
        .type   _start,@function
_start:
        call    test
	ret
end:
        .size   _start, end-_start
        .section        ".note.GNU-split-stack","",@progbits


#--- callee.s
        .globl  test
        .type   test,@function
test:
	ret
