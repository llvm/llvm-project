# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld --shared %t.o -o /dev/null 2>&1 | count 0
# RUN: ld.lld -e A --unresolved-symbols=ignore-all %t.o -o /dev/null 2>&1 | count 0
    .section    .text.A,"ax",@progbits
    .globl  A
A:
    callq B

    .cg_profile A, B, 10
