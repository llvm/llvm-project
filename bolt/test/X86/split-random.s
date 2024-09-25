## Test random function splitting option

# RUN: llvm-mc --filetype=obj --triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.random2 --split-functions \
# RUN:         --split-strategy=random2 --print-finalized \
# RUN:         --print-only=two_block --bolt-seed=7 2>&1 | \
# RUN:     FileCheck %s
# RUN: llvm-bolt %t.exe -o %t.randomN --split-functions \
# RUN:         --split-strategy=randomN --print-finalized \
# RUN:         --print-only=two_block --bolt-seed=7 2>&1 | \
# RUN:     FileCheck %s

# CHECK: Binary Function "two_block"
# CHECK:   IsSplit     :
# CHECK-SAME: {{ 1$}}

        .text
        .globl  single_block
        .type   single_block, @function
single_block:
        ret
        .size   single_block, .-single_block


        .globl  two_block
        .type   two_block, @function
two_block:
.L3:
        subl    $1, %edi
        testl   %edi, %edi
        jg      .L3
        jmp     .L4
.L4:
        subl    $1, %edi
        subl    $1, %edi
        subl    $1, %edi
        subl    $1, %edi
        ret
        .size   two_block, .-two_block


        .globl  main
        .type   main, @function
main:
        call    single_block
        movl    $10, %edi
        call    two_block
        ret
        .size   main, .-main
