## Check that ICF does not fold functions with identical code but different
## call frame information. A CFI pseudo instruction only holds an index into
## the frame instruction table of its function, and the rules in effect at the
## entry (CIE initial instructions plus FDE instructions before the first
## advance) are not in the instruction stream.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --icf=all 2>&1 | \
# RUN:   FileCheck %s --check-prefix=CHECK-ICF
# RUN: llvm-nm -n %t.bolt > %t.txt
# RUN: llvm-dwarfdump --eh-frame %t.bolt >> %t.txt
# RUN: FileCheck %s --input-file %t.txt

## Only same_cfi_b is folded with same_cfi_a.
# CHECK-ICF: BOLT-INFO: ICF folded 1 out of {{[0-9]+}} functions

# CHECK-DAG: [[#%x,BODY_B:]] T body_cfi_b
# CHECK-DAG: [[#%x,ENTRY_B:]] T entry_cfi_b
# CHECK-DAG: [[#%x,COLD_B:]] t hot_big.cold
# CHECK-DAG: [[#%x,SAME:]] T same_cfi_a
# CHECK-DAG: [[#SAME]] T same_cfi_b

# CHECK: .eh_frame contents:

## body_cfi_b keeps its own rules after the push.
# CHECK-DAG: 0x[[#%x,BODY_B+1]]: CFA=RSP+32: RBP=[CFA-24], RIP=[CFA-8]

## entry_cfi_b marks the outermost frame from its first instruction.
# CHECK-DAG: 0x[[#%x,ENTRY_B]]: CFA=RSP+8: RIP=undefined

## The cold fragment of hot_big is entered with the frame of hot_big.
# CHECK-DAG: 0x[[#%x,COLD_B]]: CFA=RSP+272: RIP=[CFA-8]

  .text
  .globl _start
  .type _start, %function
_start:
  call same_cfi_a
  call same_cfi_b
  call body_cfi_a
  call body_cfi_b
  call entry_cfi_a
  call entry_cfi_b
  call hot_small
  call hot_big
  xorl %edi, %edi
  movl $60, %eax
  syscall
  .size _start, .-_start

  .globl fail
  .type fail, %function
fail:
  ud2
  .size fail, .-fail

## Identical code and identical CFI: folded.
  .globl same_cfi_a
  .type same_cfi_a, %function
same_cfi_a:
  .cfi_startproc
  pushq %rbx
  .cfi_def_cfa_offset 16
  .cfi_offset %rbx, -16
  movl $1, %eax
  popq %rbx
  .cfi_def_cfa_offset 8
  retq
  .cfi_endproc
  .size same_cfi_a, .-same_cfi_a

  .globl same_cfi_b
  .type same_cfi_b, %function
same_cfi_b:
  .cfi_startproc
  pushq %rbx
  .cfi_def_cfa_offset 16
  .cfi_offset %rbx, -16
  movl $1, %eax
  popq %rbx
  .cfi_def_cfa_offset 8
  retq
  .cfi_endproc
  .size same_cfi_b, .-same_cfi_b

## Identical code and CFI pseudos at the same positions with the same indices,
## but different frame instructions behind them.
  .globl body_cfi_a
  .type body_cfi_a, %function
body_cfi_a:
  .cfi_startproc
  pushq %rbx
  .cfi_def_cfa_offset 16
  .cfi_offset %rbx, -16
  movl $2, %eax
  popq %rbx
  .cfi_def_cfa_offset 8
  retq
  .cfi_endproc
  .size body_cfi_a, .-body_cfi_a

  .globl body_cfi_b
  .type body_cfi_b, %function
body_cfi_b:
  .cfi_startproc
  pushq %rbx
  .cfi_def_cfa_offset 32
  .cfi_offset %rbp, -24
  movl $2, %eax
  popq %rbx
  .cfi_def_cfa_offset 8
  retq
  .cfi_endproc
  .size body_cfi_b, .-body_cfi_b

## Identical code, different rules at the entry.
  .globl entry_cfi_a
  .type entry_cfi_a, %function
entry_cfi_a:
  .cfi_startproc
  xorl %ebp, %ebp
  movq %rsp, %rdi
  call fail
  .cfi_endproc
  .size entry_cfi_a, .-entry_cfi_a

  .globl entry_cfi_b
  .type entry_cfi_b, %function
entry_cfi_b:
  .cfi_startproc
  .cfi_undefined %rip
  xorl %ebp, %ebp
  movq %rsp, %rdi
  call fail
  .cfi_endproc
  .size entry_cfi_b, .-entry_cfi_b

## Cold fragments as emitted by GCC: identical code, but each one is entered
## with the frame of its parent.
  .globl hot_small
  .type hot_small, %function
hot_small:
  .cfi_startproc
  subq $24, %rsp
  .cfi_def_cfa_offset 32
  cmpl $42, %edi
  je hot_small.cold
  addq $24, %rsp
  .cfi_def_cfa_offset 8
  retq
  .cfi_endproc
  .size hot_small, .-hot_small

  .globl hot_big
  .type hot_big, %function
hot_big:
  .cfi_startproc
  subq $264, %rsp
  .cfi_def_cfa_offset 272
  cmpl $42, %edi
  je hot_big.cold
  addq $264, %rsp
  .cfi_def_cfa_offset 8
  retq
  .cfi_endproc
  .size hot_big, .-hot_big

  .section .text.unlikely,"ax",@progbits
  .type hot_small.cold, %function
hot_small.cold:
  .cfi_startproc
  .cfi_def_cfa_offset 32
  call fail
  .cfi_endproc
  .size hot_small.cold, .-hot_small.cold

  .type hot_big.cold, %function
hot_big.cold:
  .cfi_startproc
  .cfi_def_cfa_offset 272
  call fail
  .cfi_endproc
  .size hot_big.cold, .-hot_big.cold
