## Test BinaryContext::calculateEmittedSize's functionality to update
## BinaryBasicBlock::OutputAddressRange in place so that the emitted size
## of each basic block is given by BinaryBasicBlock::getOutputSize()

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

# SPLITALL: {{^\.LBB00}}
# SPLITALL: Output Address Range: [0x0, 0x12) (18 bytes)
# SPLITALL: {{^\.LFT0}}
# SPLITALL: Output Address Range: [0x0, 0xa) (10 bytes)
# SPLITALL: {{^\.Ltmp1}}
# SPLITALL: Output Address Range: [0x0, 0x2) (2 bytes)
# SPLITALL: {{^\.Ltmp0}}
# SPLITALL: Output Address Range: [0x0, 0x10) (16 bytes)
# SPLITALL: {{^\.Ltmp2}}
# SPLITALL: Output Address Range: [0x0, 0x8) (8 bytes)
# SPLITALL: {{^\.LFT1}}
# SPLITALL: Output Address Range: [0x0, 0x8) (8 bytes)

# SPLITHOTCOLD: {{^\.LBB00}}
# SPLITHOTCOLD: Output Address Range: [0x0, 0x9) (9 bytes)
# SPLITHOTCOLD: {{^\.LFT0}}
# SPLITHOTCOLD: Output Address Range: [0x9, 0xe) (5 bytes)
# SPLITHOTCOLD: {{^\.Ltmp1}}
# SPLITHOTCOLD: Output Address Range: [0xe, 0x10) (2 bytes)
# SPLITHOTCOLD: {{^\.Ltmp0}}
# SPLITHOTCOLD: Output Address Range: [0x10, 0x1b) (11 bytes)
# SPLITHOTCOLD: {{^\.Ltmp2}}
# SPLITHOTCOLD: Output Address Range: [0x1b, 0x20) (5 bytes)
# SPLITHOTCOLD: {{^\.LFT1}}
# SPLITHOTCOLD: Output Address Range: [0x0, 0x8) (8 bytes)

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
