## Tests TailDuplication for moderate (--tail-duplication=moderate) and aggressive (--tail-duplication=aggressive) modes.
## The test uses a hot edge to a tail block and checks that a duplicated tail block is created and has the correct predecessor.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: %clang %cflags --target=aarch64-unknown-linux %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe --data %t.fdata --reorder-blocks=none \
# RUN:    --print-finalized --tail-duplication=moderate \
# RUN:    --tail-duplication-minimum-offset=1 -o %t.out | FileCheck %s
# RUN: llvm-bolt %t.exe --data %t.fdata --reorder-blocks=none \
# RUN:    --print-finalized --tail-duplication=aggressive \
# RUN:    --tail-duplication-minimum-offset=1 -o %t.out \
# RUN:    | FileCheck %s --check-prefix=CHECK-NOLOOP

# FDATA: 1 main 4 1 main #.BB2# 0 10
# CHECK: BOLT-INFO: tail duplication modified 1 ({{.*}}%) functions; duplicated 1 blocks (4 bytes) responsible for {{.*}} dynamic executions ({{.*}}% of all block executions)
# CHECK: BB Layout   : .LBB00, .Ltail-dup0, .Ltmp0, .Ltmp1
# CHECK-LABEL: .LBB00 (1 instructions, align : 1)
# CHECK-NEXT:   Entry Point
# CHECK-NEXT:   Exec Count : {{.*}}
# CHECK-NEXT:   eor w0, w0, w0
# CHECK-NEXT:   Successors: .Ltail-dup0

## Check that the predecessor of Ltail-dup0 is .LBB00, not itself.
# CHECK-NOLOOP-LABEL: .Ltail-dup0 (1 instructions, align : 1)
# CHECK-NOLOOP-NEXT:   Exec Count : {{.*}}
# CHECK-NOLOOP-NEXT:   Predecessors: .LBB00
# CHECK-NOLOOP-NEXT:   ret
    .text
    .globl main
    .type main, %function
main:
.BB0:
    eor w0, w0, w0
    b .BB2
.BB1:
    add x0, x0, #1
.BB2:
    ret

## Force relocation mode against text.
    .reloc 0, R_AARCH64_NONE
.Lend:
