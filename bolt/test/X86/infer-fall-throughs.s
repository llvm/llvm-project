## Test that infer-fall-throughs would correctly infer the wrong fall-through
## edge count in the example

# RUN: llvm-mc --filetype=obj --triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt \
# RUN:         --print-estimate-edge-counts --data=%t.fdata \
# RUN:     2>&1 | FileCheck --check-prefix=WITHOUTINFERENCE %s
# RUN: llvm-bolt %t.exe -o %t.bolt --infer-fall-throughs \
# RUN:         --print-estimate-edge-counts --data=%t.fdata \
# RUN:     2>&1 | FileCheck --check-prefix=CORRECTINFERENCE %s


# WITHOUTINFERENCE: Binary Function "main" after estimate-edge-counts
# WITHOUTINFERENCE: {{^\.Ltmp0}}
# WITHOUTINFERENCE: Successors: .Ltmp1 (mispreds: 0, count: 10), .LFT0 (mispreds: 0, count: 0)
# WITHOUTINFERENCE: {{^\.LFT0}}
# WITHOUTINFERENCE: Exec Count : 490

# CORRECTINFERENCE: Binary Function "main" after estimate-edge-counts
# CORRECTINFERENCE: {{^\.Ltmp0}}
# CORRECTINFERENCE: Successors: .Ltmp1 (mispreds: 0, count: 10), .LFT0 (inferred count: 490)
# CORRECTINFERENCE: {{^\.LFT0}}
# CORRECTINFERENCE: Exec Count : 490


        .globl  main
        .type   main, @function
main:
LLmain_LLstart:
        jmp     LLstart
# FDATA: 1 main #LLmain_LLstart# 1 main #LLstart# 0 500
LLstart:
        jge     LLexit
# FDATA: 1 main #LLstart# 1 main #LLexit# 0 10
# FDATA: 1 main #LLstart# 1 main #LLmore# 0 0
LLmore:
        movl    $5, %eax
# FDATA: 1 main #LLmore# 1 main #LLexit# 0 490
LLexit:
        ret
.LLmain_end:
        .size   main, .LLmain_end-main
