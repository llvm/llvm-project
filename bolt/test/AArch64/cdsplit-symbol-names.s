## Test the correctness of section names and function symbol names post cdsplit.
 
 
# RUN: llvm-mc --filetype=obj --triple aarch64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-strategy=cdsplit  \
# RUN:         --data=%t.fdata --reorder-blocks=ext-tsp  
# RUN: llvm-objdump --syms %t.bolt | FileCheck %s --check-prefix=CHECK-SYMS-WARM
 
# CHECK-SYMS-WARM: 0000000000000000 l df *ABS* 0000000000000000 bolt-pseudo.o
# CHECK-SYMS-WARM: .text.cold
# CHECK-SYMS-WARM-SAME: chain.cold
# CHECK-SYMS-WARM: .text.warm
# CHECK-SYMS-WARM-SAME: chain.warm
 
        .section .text
        .globl  chain
        .type   chain, %function
chain:
        stp     x29, x30, [sp, #-16]!  
        mov     x29, sp                
        cmp     w0, #2
LLentry_LLchain_start:
        b.ge    LLchain_start
# FDATA: 1 chain #LLentry_LLchain_start# 1 chain #LLchain_start# 0 10
# FDATA: 1 chain #LLentry_LLchain_start# 1 chain #LLfast# 0 500
LLfast:
        mov     w0, #5
LLfast_LLexit:
        b       LLexit
# FDATA: 1 chain #LLfast_LLexit# 1 chain #LLexit# 0 500
LLchain_start:
        mov     w0, #10
LLchain_start_LLchain1:
        b.ge    LLchain1
# FDATA: 1 chain #LLchain_start_LLchain1# 1 chain #LLchain1# 0 10
# FDATA: 1 chain #LLchain_start_LLchain1# 1 chain #LLcold# 0 0
LLcold:
        add     w0, w0, #1
        add     w0, w0, #1
        add     w0, w0, #1
        add     w0, w0, #1
        add     w0, w0, #1
        add     w0, w0, #1
LLchain1:
        add     w0, w0, #1
LLchain1_LLchain2:
        b       LLchain2
# FDATA: 1 chain #LLchain1_LLchain2# 1 chain #LLchain2# 0 10
LLchain2:
        add     w0, w0, #1
LLchain2_LLchain3:
        b       LLchain3
# FDATA: 1 chain #LLchain2_LLchain3# 1 chain #LLchain3# 0 10
LLchain3:
        add     w0, w0, #1
        add     w0, w0, #1
        add     w0, w0, #1
        add     w0, w0, #1
        add     w0, w0, #1
LLchain3_LLchain4:
        b       LLchain4
# FDATA: 1 chain #LLchain3_LLchain4# 1 chain #LLchain4# 0 10
LLchain4:
        add     w0, w0, #1
        add     w0, w0, #1
        add     w0, w0, #1
        add     w0, w0, #1
        add     w0, w0, #1
LLchain4_LLexit:
        b       LLexit
# FDATA: 1 chain #LLchain4_LLexit# 1 chain #LLexit# 0 10
LLexit:
        ldp     x29, x30, [sp], #16   
        ret
LLchain_end:
        .size   chain, LLchain_end-chain
 
        .globl  main
        .type   main, %function
main:
        stp     x29, x30, [sp, #-16]!  
        mov     x29, sp                
        mov     w0, #1                 
LLmain_chain1:
        bl      chain
# FDATA: 1 main #LLmain_chain1# 1 chain 0 0 500
        mov     w0, #4                 
LLmain_chain2:
        bl      chain
# FDATA: 1 main #LLmain_chain2# 1 chain 0 0 10
        mov     w0, #0                 
        ldp     x29, x30, [sp], #16    
        ret
.Lmain_end:
        .size   main, .Lmain_end-main