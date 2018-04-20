# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

# RUN: echo "A B 100" > %t.call_graph
# RUN: echo "A C 40" >> %t.call_graph
# RUN: echo "B C 30" >> %t.call_graph
# RUN: echo "adena A 30" >> %t.call_graph
# RUN: echo "poppy A 30" >> %t.call_graph
# RUN: ld.lld -e A %t --call-graph-ordering-file %t.call_graph -o %t.out \
# RUN:   -noinhibit-exec -icf=all 2>&1 | FileCheck %s

    .section    .text.C,"ax",@progbits
    .globl  C
C:
    mov poppy, %rax
    retq

B = 0x1234

    .section    .text.A,"ax",@progbits
    .globl  A
A:
    mov poppy, %rax
    retq

# CHECK: unable to order absolute symbol: B
# CHECK: call graph file: no such symbol: adena
# CHECK: unable to order undefined symbol: poppy
