## Check that edge weights are correctly distributed among CFG edges being
## estimated by BOLT when TotalChildrenCount is zero and non-zero.

# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-unknown %s -o %t.o
# RUN: link_fdata --no-lbr %s %t.o %t.z.fdata FDATA_ZERO
# RUN: link_fdata --no-lbr %s %t.o %t.nz.fdata FDATA_NONZERO
# RUN: %clang %cflags -Wl,-q %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.z.bolt --print-estimate-edge-counts \
# RUN:   --data=%t.z.fdata 2>&1 | FileCheck %s --check-prefix=ZERO
# RUN: llvm-bolt %t.exe -o %t.nz.bolt --print-estimate-edge-counts \
# RUN:   --data=%t.nz.fdata 2>&1 | FileCheck %s --check-prefix=NONZERO

# ZERO-LABEL: Binary Function "main" after estimate-edge-counts
# ZERO-LABEL: .LBB00 (
# ZERO: Successors: .Ltmp0 (mispreds: 0, count: 3000)
# ZERO-LABEL: .Ltmp1 (
# ZERO: Successors: .Ltmp0 (mispreds: 0, count: 3000)

# FDATA_ZERO: 1 main #LBB0_1# 6

# NONZERO-LABEL: Binary Function "main" after estimate-edge-counts
# NONZERO-LABEL: .LBB00 (
# NONZERO: Successors: .Ltmp0 (mispreds: 0, count: 1000)
# NONZERO-LABEL: .Ltmp1 (
# NONZERO: Successors: .Ltmp0 (mispreds: 0, count: 5000)

# FDATA_NONZERO: 1 main #main# 1
# FDATA_NONZERO: 1 main #LBB0_1# 6
# FDATA_NONZERO: 1 main #LBB0_2# 5
# FDATA_NONZERO: 1 main #LBB0_3# 1

        .file   "main.c"
        .text
        .globl  main
        .p2align        2
        .type   main,@function
main:
        sub     sp, sp, #16
        str     wzr, [sp, #12]
        mov     w8, #5
        str     w8, [sp, #8]
        b       LBB0_1
LBB0_1:
        ldr     w8, [sp, #8]
        subs    w8, w8, #0
        b.gt    LBB0_2
        b       LBB0_3
LBB0_2:
        ldr     w8, [sp, #8]
        subs    w8, w8, #1
        str     w8, [sp, #8]
        b       LBB0_1
LBB0_3:
        mov     w0, wzr
        add     sp, sp, #16
        ret
Lfunc_end0:
        .size   main, Lfunc_end0-main
        .reloc 0, R_AARCH64_NONE
        .section        ".note.GNU-stack","",@progbits
        .addrsig
