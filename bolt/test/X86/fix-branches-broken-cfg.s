## Check that fixBranches() is not invoked on a broken CFG which could lead to
## unintended consequences including a firing assertion.

# RUN: llvm-mc --filetype=obj --triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-strategy=cdsplit \
# RUN:         --data=%t.fdata --reorder-blocks=ext-tsp 2>&1 | FileCheck %s

# CHECK: internal call detected

        .text

        .globl  foo
        .type   foo, @function
foo:
        ret
        .size foo, .-foo

## main contains an internal call. ValidateInternalCalls pass will modify CFG
## (making it invalid) and mark the function as non-simple. After that, we
## cannot make any assumption about the CFG.

        .globl  main
        .type   main, @function
main:
        call .L1
        ret
.L1:
        pushq   %rbp
        movq    %rsp, %rbp
        movl    $1, %edi
LLmain_foo1:
        call    foo
# FDATA: 1 main #LLmain_foo1# 1 foo 0 0 600
        movl    $4, %edi
        xorl    %eax, %eax
        popq    %rbp
        retq
.Lmain_end:
        .size   main, .Lmain_end-main
