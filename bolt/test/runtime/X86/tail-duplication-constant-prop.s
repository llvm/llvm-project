# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe --data %t.fdata --reorder-blocks=ext-tsp \
# RUN:    --print-finalized \
# RUN:    --tail-duplication=moderate --tail-duplication-minimum-offset=1 \
# RUN:    --tail-duplication-const-copy-propagation=1 -o %t.out | FileCheck %s
# RUN: %t.exe; echo $?
# RUN: %t.out; echo $?

# FDATA: 1 main 14 1 main #.BB2# 0 10
# FDATA: 1 main 16 1 main #.BB2# 0 20
# CHECK: BOLT-INFO: tail duplication modified 1 ({{.*}}%) functions; duplicated 1 blocks ({{.*}} bytes) responsible for {{.*}} dynamic executions ({{.*}}% of all block executions)
# CHECK: BB Layout   : .LBB00, .Ltail-dup0, .Ltmp0, .Ltmp1
# CHECK-NOT: mov $0x2, %rbx

    .text
    .globl main
    .type main, %function
    .size main, .Lend-main
main:
    mov $0x2, %rbx
    mov $0x1, %rdi
    inc %rdi
    mov %rdi, %rsi
    jmp .BB2
.BB1:
    mov $0x9, %rbx
.BB2:
    mov %rbx, %rax
    mov $0x5, %rbx
    add %rsi, %rax
    retq
.BB3:
    mov $0x9, %rbx
.BB4:
    mov $0xa, %rsi
    add %rbx, %rax
    add %rsi, %rax
.BB5:
    retq
.Lend:
