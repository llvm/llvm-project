# Test BinaryContext::calculateEmittedSize's functionality to update
# BinaryBasicBlock::OutputAddressRange in place so that the emitted size
# of each basic block is given by BinaryBasicBlock::getOutputSize()

# RUN: llvm-mc --filetype=obj --triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-strategy=all \
# RUN:         --print-split --print-only=chain --print-output-address-range \
# RUN:         --data=%t.fdata --reorder-blocks=ext-tsp \
# RUN:     2>&1 | FileCheck --check-prefix=SPLITALL %s
# RUN: llvm-mc --filetype=obj --triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --print-split \
# RUN:         --print-only=chain --print-output-address-range \
# RUN:         --data=%t.fdata --reorder-blocks=ext-tsp \
# RUN:     2>&1 | FileCheck --check-prefix=SPLITHOTCOLD %s

# SPLITALL: Binary Function "chain" after split-functions
# SPLITALL: {{^\.LBB00}}
# SPLITALL: Output Start Address: 0
# SPLITALL: Output End Address: 18
# SPLITALL: {{^\.LFT0}}
# SPLITALL: Output Start Address: 0
# SPLITALL: Output End Address: 10
# SPLITALL: {{^\.Ltmp1}}
# SPLITALL: Output Start Address: 0
# SPLITALL: Output End Address: 2
# SPLITALL: {{^\.Ltmp0}}
# SPLITALL: Output Start Address: 0
# SPLITALL: Output End Address: 16
# SPLITALL: {{^\.Ltmp2}}
# SPLITALL: Output Start Address: 0
# SPLITALL: Output End Address: 8
# SPLITALL: {{^\.LFT1}}
# SPLITALL: Output Start Address: 0
# SPLITALL: Output End Address: 8

# SPLITHOTCOLD: {{^\.LBB00}}
# SPLITHOTCOLD: Output Start Address: 0
# SPLITHOTCOLD: Output End Address: 9
# SPLITHOTCOLD: {{^\.LFT0}}
# SPLITHOTCOLD: Output Start Address: 9
# SPLITHOTCOLD: Output End Address: 14
# SPLITHOTCOLD: {{^\.Ltmp1}}
# SPLITHOTCOLD: Output Start Address: 14
# SPLITHOTCOLD: Output End Address: 16
# SPLITHOTCOLD: {{^\.Ltmp0}}
# SPLITHOTCOLD: Output Start Address: 16
# SPLITHOTCOLD: Output End Address: 27
# SPLITHOTCOLD: {{^\.Ltmp2}}
# SPLITHOTCOLD: Output Start Address: 27
# SPLITHOTCOLD: Output End Address: 32
# SPLITHOTCOLD: {{^\.LFT1}}
# SPLITHOTCOLD: Output Start Address: 0
# SPLITHOTCOLD: Output End Address: 8

        .text
        .globl  chain
        .type   chain, @function
chain:
        pushq   %rbp
        movq    %rsp, %rbp
        cmpl    $2, %edi
LLentry_LLchain_start:
        jge     LLchain_start
# FDATA: 1 chain #LLentry_LLchain_start# 1 chain #LLchain_start# 0 10
# FDATA: 1 chain #LLentry_LLchain_start# 1 chain #LLfast# 0 500
LLfast:
        movl    $5, %eax
LLfast_LLexit:
        jmp     LLexit
# FDATA: 1 chain #LLfast_LLexit# 1 chain #LLexit# 0 500
LLchain_start:
        movl    $10, %eax
LLchain_start_LLchain1:
        jge     LLchain1
# FDATA: 1 chain #LLchain_start_LLchain1# 1 chain #LLchain1# 0 10
# FDATA: 1 chain #LLchain_start_LLchain1# 1 chain #LLcold# 0 0
LLcold:
        addl    $1, %eax
LLchain1:
        addl    $1, %eax
LLchain1_LLexit:
        jmp     LLexit
# FDATA: 1 chain #LLchain1_LLexit# 1 chain #LLexit# 0 10
LLexit:
        popq    %rbp
        ret
LLchain_end:
        .size   chain, LLchain_end-chain


        .globl  main
        .type   main, @function
main:
        pushq   %rbp
        movq    %rsp, %rbp
        movl    $1, %edi
LLmain_chain1:
        call    chain
# FDATA: 1 main #LLmain_chain1# 1 chain 0 0 500
        movl    $4, %edi
LLmain_chain2:
        call    chain
# FDATA: 1 main #LLmain_chain2# 1 chain 0 0 10
        xorl    %eax, %eax
        popq    %rbp
        retq
.Lmain_end:
        .size   main, .Lmain_end-main
