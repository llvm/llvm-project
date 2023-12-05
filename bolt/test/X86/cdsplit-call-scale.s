# Test the control of aggressiveness of 3-way splitting by -call-scale.
# When -call-scale=0.0, the tested function is 2-way splitted.
# When -call-scale=1.0, the tested function is 3-way splitted with 5 blocks
# in warm because of the increased benefit of shortening the call edges.
# When -call-scale=1000.0, the tested function is 3-way splitted with 7 blocks
# in warm because of the strong benefit of shortening the call edges.

# RUN: llvm-mc --filetype=obj --triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-strategy=cdsplit \
# RUN:         --call-scale=0.0 --print-split --print-only=chain \
# RUN:         --data=%t.fdata --reorder-blocks=ext-tsp \
# RUN:     2>&1 | FileCheck --check-prefix=LOWINCENTIVE %s
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-strategy=cdsplit \
# RUN:         --call-scale=1.0 --print-split --print-only=chain \
# RUN:         --data=%t.fdata --reorder-blocks=ext-tsp \
# RUN:     2>&1 | FileCheck --check-prefix=MEDINCENTIVE %s
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-strategy=cdsplit \
# RUN:         --call-scale=1000.0 --print-split --print-only=chain \
# RUN:         --data=%t.fdata --reorder-blocks=ext-tsp \
# RUN:     2>&1 | FileCheck --check-prefix=HIGHINCENTIVE %s

# LOWINCENTIVE: Binary Function "chain" after split-functions
# LOWINCENTIVE: {{^\.Ltmp5}}
# LOWINCENTIVE: -------   HOT-COLD SPLIT POINT   -------
# LOWINCENTIVE: {{^\.LFT1}}

# MEDINCENTIVE: Binary Function "chain" after split-functions
# MEDINCENTIVE: {{^\.Ltmp1}}
# MEDINCENTIVE: -------   HOT-COLD SPLIT POINT   -------
# MEDINCENTIVE: {{^\.LFT1}}
# MEDINCENTIVE: -------   HOT-COLD SPLIT POINT   -------
# MEDINCENTIVE: {{^\.Ltmp0}}
# MEDINCENTIVE: {{^\.Ltmp2}}
# MEDINCENTIVE: {{^\.Ltmp3}}
# MEDINCENTIVE: {{^\.Ltmp4}}
# MEDINCENTIVE: {{^\.Ltmp5}}

# HIGHINCENTIVE: Binary Function "chain" after split-functions
# HIGHINCENTIVE: {{^\.LBB00}}
# HIGHINCENTIVE: -------   HOT-COLD SPLIT POINT   -------
# HIGHINCENTIVE: {{^\.LFT1}}
# HIGHINCENTIVE: -------   HOT-COLD SPLIT POINT   -------
# HIGHINCENTIVE: {{^\.LFT0}}
# HIGHINCENTIVE: {{^\.Ltmp1}}
# HIGHINCENTIVE: {{^\.Ltmp0}}
# HIGHINCENTIVE: {{^\.Ltmp2}}
# HIGHINCENTIVE: {{^\.Ltmp3}}
# HIGHINCENTIVE: {{^\.Ltmp4}}
# HIGHINCENTIVE: {{^\.Ltmp5}}



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
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
LLchain1:
        addl    $1, %eax
LLchain1_LLchain2:
        jmp     LLchain2
# FDATA: 1 chain #LLchain1_LLchain2# 1 chain #LLchain2# 0 10
LLchain2:
        addl    $1, %eax
LLchain2_LLchain3:
        jmp     LLchain3
# FDATA: 1 chain #LLchain2_LLchain3# 1 chain #LLchain3# 0 10
LLchain3:
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
LLchain3_LLchain4:
        jmp     LLchain4
# FDATA: 1 chain #LLchain3_LLchain4# 1 chain #LLchain4# 0 10
LLchain4:
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
LLchain4_LLexit:
        jmp     LLexit
# FDATA: 1 chain #LLchain4_LLexit# 1 chain #LLexit# 0 10
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
