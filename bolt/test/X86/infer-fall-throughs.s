## Test that infer-fall-throughs would correctly infer the wrong fall-through
## edge count in the example

# RUN: llvm-mc --filetype=obj --triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions \
# RUN:         --print-split --print-only=chain --data=%t.fdata \
# RUN:         --reorder-blocks=none \
# RUN:     2>&1 | FileCheck --check-prefix=WITHOUTINFERENCE %s
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --infer-fall-throughs \
# RUN:         --print-split --print-only=chain --data=%t.fdata \
# RUN:         --reorder-blocks=none \
# RUN:     2>&1 | FileCheck --check-prefix=CORRECTINFERENCE %s


# WITHOUTINFERENCE: Binary Function "chain" after split-functions
# WITHOUTINFERENCE: {{^\.LBB00}}
# WITHOUTINFERENCE: Successors: .Ltmp0 (mispreds: 0, count: 10), .LFT0 (mispreds: 0, count: 0)
# WITHOUTINFERENCE: {{^\.LFT0}}
# WITHOUTINFERENCE: Exec Count : 490

# CORRECTINFERENCE: Binary Function "chain" after split-functions
# CORRECTINFERENCE: {{^\.LBB00}}
# CORRECTINFERENCE: Successors: .Ltmp0 (mispreds: 0, count: 10), .LFT0 (inferred count: 490)
# CORRECTINFERENCE: {{^\.LFT0}}
# CORRECTINFERENCE: Exec Count : 490


        .text
        .globl  chain
        .type   chain, @function
chain:
        pushq   %rbp
        movq    %rsp, %rbp
        cmpl    $2, %edi
LLstart:
        jge     LLless
# FDATA: 1 chain #LLstart# 1 chain #LLless# 0 10
# FDATA: 1 chain #LLstart# 1 chain #LLmore# 0 0
LLmore:
        movl    $5, %eax
LLmore_LLexit:
        jmp     LLexit
# FDATA: 1 chain #LLmore_LLexit# 1 chain #LLexit# 0 490
LLless:
        movl    $10, %eax
LLdummy:
        jmp     LLexit
# FDATA: 1 chain #LLdummy# 1 chain #LLexit# 0 10
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
LLmain_chain:
        call    chain
# FDATA: 1 main #LLmain_chain# 1 chain 0 0 500
        movl    $4, %edi
        retq
.Lmain_end:
        .size   main, .Lmain_end-main
