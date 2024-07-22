## Test the correctness of section names and function symbol names post cdsplit.
## Warm section should have name .text.warm and warm function fragments should
## have symbol names ending in warm.

# RUN: llvm-mc --filetype=obj --triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-strategy=cdsplit \
# RUN:         --call-scale=2 --data=%t.fdata --reorder-blocks=ext-tsp
# RUN: llvm-objdump --syms %t.bolt | FileCheck %s --check-prefix=CHECK-SYMS-WARM

# CHECK-SYMS-WARM: 0000000000000000 l df *ABS* 0000000000000000 bolt-pseudo.o
# CHECK-SYMS-WARM: .text.warm
# CHECK-SYMS-WARM-SAME: chain.warm
# CHECK-SYMS-WARM: .text.cold
# CHECK-SYMS-WARM-SAME: dummy.cold

        .text
        .globl  chain
        .type   chain, @function
chain:
        pushq   %rbp
        movq    %rsp, %rbp
        cmpl    $2, %edi
LLentry_LLchain_start:
        jge     LLchain_start
# FDATA: 1 chain #LLentry_LLchain_start# 1 chain #LLchain_start# 0 100
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
# FDATA: 1 chain #LLchain_start_LLchain1# 1 chain #LLchain1# 0 99
# FDATA: 1 chain #LLchain_start_LLchain1# 1 chain #LLloop_entry# 0 1
LLloop_entry:
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        xorl    %eax, %eax          // Initialize result
        movl    $1000000, %ecx      // Set loop counter to a large value
LLloop_entry_LLloop_start:
        jmp     LLloop_start
# FDATA: 1 chain #LLloop_entry_LLloop_start# 1 chain #LLloop_start# 0 1
LLloop_start:
        addl    $1, %eax            // Increment result
        subl    $1, %ecx            // Decrement loop counter
LLloop_start_LLloop_start:
        jg      LLloop_start        // Jump if loop counter is greater than 0
# FDATA: 1 chain #LLloop_start_LLloop_start# 1 chain #LLloop_start# 0 1000000
# FDATA: 1 chain #LLloop_start_LLloop_start# 1 chain #LLchain1# 0 1
LLchain1:
        addl    $1, %eax
LLchain1_LLchain2:
        jmp     LLchain2
# FDATA: 1 chain #LLchain1_LLchain2# 1 chain #LLchain2# 0 100
LLchain2:
        addl    $1, %eax
LLchain2_LLchain3:
        jmp     LLchain3
# FDATA: 1 chain #LLchain2_LLchain3# 1 chain #LLchain3# 0 100
LLchain3:
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
LLchain3_LLchain4:
        jmp     LLchain4
# FDATA: 1 chain #LLchain3_LLchain4# 1 chain #LLchain4# 0 100
LLchain4:
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
LLchain4_LLexit:
        jmp     LLexit
# FDATA: 1 chain #LLchain4_LLexit# 1 chain #LLexit# 0 100
LLexit:
        popq    %rbp
        ret
LLchain_end:
        .size   chain, LLchain_end-chain

        .text
        .globl  dummy
        .type   dummy, @function
dummy:
        pushq   %rbp
        movq    %rsp, %rbp
        cmpl    $2, %edi
dummy_dummy_block1:
        jg     dummy_block1
# FDATA: 1 dummy #dummy_dummy_block1# 1 dummy #dummy_block1# 0 0
# FDATA: 1 dummy #dummy_dummy_block1# 1 dummy #dummy_next# 0 100
dummy_next:
        addl    $1, %eax
        addl    $1, %eax
dummy_next_dummy_exit:
        jmp     dummy_exit
# FDATA: 1 dummy #dummy_next_dummy_exit# 1 dummy #dummy_exit# 0 100
dummy_block1:
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
dummy_block1_dummy_block2:
        jmp     dummy_block2
# FDATA: 1 dummy #dummy_block1_dummy_block2# 1 dummy #dummy_block2# 0 0
dummy_block2:
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
dummy_block2_dummy_exit:
        jmp     dummy_exit
# FDATA: 1 dummy #dummy_block2_dummy_exit# 1 dummy #dummy_exit# 0 0
dummy_exit:
        popq    %rbp
        ret

        .globl  main
        .type   main, @function
main:
        pushq   %rbp
        movq    %rsp, %rbp
        movl    $1, %edi
LLmain_chain1:
        call    chain
# FDATA: 1 main #LLmain_chain1# 1 chain 0 0 600
        movl    $4, %edi
LLmain_dummy:
        call    dummy
# FDATA: 1 main #LLmain_dummy# 1 dummy 0 0 100
        xorl    %eax, %eax
        popq    %rbp
        retq
.Lmain_end:
        .size   main, .Lmain_end-main
