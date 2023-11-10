# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: link_fdata %s %t.o %t.fdata2 "FDATA2"
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe --data %t.fdata --reorder-blocks=none \
# RUN:    --print-finalized --tail-duplication=cache -o %t.out | FileCheck %s
# RUN: llvm-bolt %t.exe --data %t.fdata2 --reorder-blocks=none \
# RUN:    --print-finalized --tail-duplication=cache -o %t.out2 \
# RUN:    | FileCheck --check-prefix="CHECK2" %s

# A test where the tail is duplicated to eliminate an unconditional jump
# FDATA: 1 main #.BB0_br# 1 main #.BB4# 0 100
# FDATA: 1 main #.BB0_br# 1 main #.BB1# 0 100
# FDATA: 1 main #.BB1_br# 1 main #.BB3# 0 50
# FDATA: 1 main #.BB1_br# 1 main #.BB2# 0 50
# FDATA: 1 main #.BB3_br# 1 main #.BB2# 0 50
# CHECK: BOLT-INFO: tail duplication modified 1 ({{.*}}%) functions; duplicated 1 blocks (13 bytes) responsible for 50 dynamic executions ({{.*}}% of all block executions)
# CHECK: BB Layout   : .LBB00, .Ltmp0, .Ltmp1, .Ltmp2, .Ltmp3, .Ltmp4, .Ltmp5, .Ltail-dup0, .Ltmp6

# A test where the tail is not duplicated due to the cache score
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
    xor %eax, %eax
  	cmpl	%eax, %ebx
.BB0_br:
    je      .BB4
.BB1:
    inc %rax
.BB1_br:
    je .BB3
.BB2:
    inc %rax
    inc %rax
    inc %rax
    inc %rax
    retq
.BB3:
    inc %rax
.BB3_br:
    jmp .BB2
.BB4:
    retq
# For relocations against .text
    call exit
.Lend:
