## Tests TailDuplication cache mode (--tail-duplication=cache) with two profiles:
## one that triggers duplication and one that does not.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: link_fdata %s %t.o %t.fdata2 "FDATA2"
# RUN: %clang %cflags --target=aarch64-unknown-linux %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe --data %t.fdata --reorder-blocks=none \
# RUN:    --print-finalized --tail-duplication=cache -o %t.out | FileCheck %s
# RUN: llvm-bolt %t.exe --data %t.fdata2 --reorder-blocks=none \
# RUN:    --print-finalized --tail-duplication=cache -o %t.out2 \
# RUN:    | FileCheck %s --check-prefix="CHECK2"

## A test where the tail is duplicated to eliminate an unconditional jump.
# FDATA: 1 main #.BB0_br# 1 main #.BB4# 0 100
# FDATA: 1 main #.BB0_br# 1 main #.BB1# 0 100
# FDATA: 1 main #.BB1_br# 1 main #.BB3# 0 50
# FDATA: 1 main #.BB1_br# 1 main #.BB2# 0 50
# FDATA: 1 main #.BB3_br# 1 main #.BB2# 0 50
# CHECK: BOLT-INFO: tail duplication modified 1 ({{.*}}%) functions; duplicated 1 blocks (20 bytes) responsible for 50 dynamic executions ({{.*}}% of all block executions)
# CHECK: BB Layout   : .LBB00, .Ltmp0, .Ltmp1, .Ltmp2, .Ltmp3, .Ltmp4, .Ltmp5, .Ltail-dup0, .Ltmp6
# CHECK-LABEL: .Ltail-dup0 (5 instructions, align : 1)
# CHECK-NEXT:   Exec Count : {{.*}}
# CHECK-NEXT:   Predecessors: .Ltmp5
# CHECK-NEXT:   add x0, x0, #0x1
# CHECK-NEXT:   add x0, x0, #0x1
# CHECK-NEXT:   add x0, x0, #0x1
# CHECK-NEXT:   add x0, x0, #0x1
# CHECK-NEXT:   ret

## A test where the tail is not duplicated due to the cache score.
# FDATA2: 1 main #.BB0_br# 1 main #.BB4# 0 100
# FDATA2: 1 main #.BB0_br# 1 main #.BB1# 0 2
# FDATA2: 1 main #.BB1_br# 1 main #.BB3# 0 1
# FDATA2: 1 main #.BB1_br# 1 main #.BB2# 0 1
# FDATA2: 1 main #.BB3_br# 1 main #.BB2# 0 1
# CHECK2: BOLT-INFO: tail duplication modified 0 (0.00%) functions; duplicated 0 blocks (0 bytes) responsible for 0 dynamic executions (0.00% of all block executions)
# CHECK2: BB Layout   : .LBB00, .Ltmp0, .Ltmp1, .Ltmp2, .Ltmp3, .Ltmp4, .Ltmp5, .Ltmp6

    .text
    .globl main
    .type main, %function
    .size main, .Lend-main
main:
.BB0:
    eor w0, w0, w0
.BB0_br:
    cbz w0, .BB4
.BB1:
    add x0, x0, #1
.BB1_br:
    cbz w0, .BB3
.BB2:
    add x0, x0, #1
    add x0, x0, #1
    add x0, x0, #1
    add x0, x0, #1
    ret
.BB3:
    add x0, x0, #1
.BB3_br:
    b .BB2
.BB4:
    ret

## Force relocation mode against text.
    .reloc 0, R_AARCH64_NONE
.Lend:
