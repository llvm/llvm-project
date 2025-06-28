## Test that BOLT-INFO is correctly formatted after profile quality reporting for
## a small binary.

# RUN: llvm-mc --filetype=obj --triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --data=%t.fdata \
# RUN:     2>&1 | FileCheck %s

# CHECK: BOLT-INFO: profile quality metrics for the hottest 2 functions (reporting top 5% values): function CFG discontinuity 0.00%; call graph flow conservation gap 0.00%; CFG flow conservation gap 0.00% (weighted) 0.00% (worst); exception handling usage 0.00% (of total BBEC) 0.00% (of total InvokeEC)
# CHECK-NEXT: BOLT-INFO:

        .text
        .globl  func
        .type   func, @function
func:
        pushq   %rbp
        ret
LLfunc_end:
        .size   func, LLfunc_end-func


        .globl  main
        .type   main, @function
main:
        pushq   %rbp
        movq    %rsp, %rbp
LLmain_func:
        call    func
# FDATA: 1 main #LLmain_func# 1 func 0 0 500
        movl    $4, %edi
        retq
.Lmain_end:
        .size   main, .Lmain_end-main
