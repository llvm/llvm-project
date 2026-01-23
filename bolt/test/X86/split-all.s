## Test split all block strategy

# RUN: llvm-mc --filetype=obj --triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-strategy=all \
# RUN:         --print-split --print-only=chain \
# RUN:     2>&1 | FileCheck %s

# CHECK: Binary Function "chain"
# CHECK:   IsSplit     :
# CHECK-SAME: {{ 1$}}
# CHECK: {{^\.LBB00}}
# CHECK: -------   HOT-COLD SPLIT POINT   -------
# CHECK: {{^\.LFT0}}
# CHECK: -------   HOT-COLD SPLIT POINT   -------
# CHECK: {{^\.Ltmp0}}
# CHECK: -------   HOT-COLD SPLIT POINT   -------
# CHECK: {{^\.Ltmp2}}
# CHECK: -------   HOT-COLD SPLIT POINT   -------
# CHECK: {{^\.Ltmp3}}
# CHECK: -------   HOT-COLD SPLIT POINT   -------
# CHECK: {{^\.Ltmp4}}
# CHECK: -------   HOT-COLD SPLIT POINT   -------
# CHECK: {{^\.Ltmp5}}
# CHECK: -------   HOT-COLD SPLIT POINT   -------
# CHECK: {{^\.Ltmp1}}
# CHECK: End of Function "chain"

        .text
        .globl  chain
        .type   chain, @function
chain:
.Lchain_entry:
        pushq   %rbp
        movq    %rsp, %rbp
        cmpl    $2, %edi
        jge     .Lchain_start
.Lfast:
        movl    $5, %eax
        jmp     .Lexit
.Lchain_start:
        movl    $10, %eax
        jmp     .Lchain1
.Lchain1:
        addl    $1, %eax
        jmp     .Lchain2
.Lchain2:
        addl    $1, %eax
        jmp     .Lchain3
.Lchain3:
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        jmp     .Lchain4
.Lchain4:
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        addl    $1, %eax
        jmp     .Lexit
.Lexit:
        popq    %rbp
        ret
.Lchain_end:
        .size   chain, .Lchain_end-chain


        .globl  main
        .type   main, @function
main:
        pushq   %rbp
        movq    %rsp, %rbp
        movl    $1, %edi
        call    chain
        movl    $4, %edi
        call    chain
        xorl    %eax, %eax
        popq    %rbp
        retq
.Lmain_end:
        .size   main, .Lmain_end-main
