# Check that BOLT in lite mode processes fragments as expected.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.out --lite=1 --data %t.fdata -v=1 2>&1 \
# RUN:   | FileCheck %s

# CHECK: BOLT-INFO: skipping processing main.cold.1 together with parent function
# CHECK: BOLT-INFO: skipping processing foo.cold.1/1 together with parent function
# CHECK: BOLT-INFO: skipping processing bar.cold.1/1 together with parent function

  .globl main
  .type main, %function
main:
  .cfi_startproc
  cmpl	$0x0, %eax
  je	main.cold.1
  retq
  .cfi_endproc
.size main, .-main

  .globl foo
  .type foo, %function
foo:
  .cfi_startproc
  cmpl	$0x0, %eax
  je	foo.cold.1
  retq
  .cfi_endproc
.size foo, .-foo

  .local bar
  .type bar, %function
bar:
  .cfi_startproc
  cmpl	$0x0, %eax
  je	bar.cold.1
  retq
  .cfi_endproc
.size bar, .-bar

  .section .text.cold
  .globl main.cold.1
  .type main.cold.1, %function
main.cold.1:
# FDATA: 0 [unknown] 0 1 main.cold.1 0 1 0
  .cfi_startproc
  pushq	%rbp
  movq	%rsp, %rbp
  movl	$0x0, %eax
  popq	%rbp
  retq
  .cfi_endproc
.size main.cold.1, .-main.cold.1

  .local foo.cold.1
  .type foo.cold.1, %function
foo.cold.1:
# FDATA: 0 [unknown] 0 1 foo.cold.1/1 0 1 0
  .cfi_startproc
  pushq	%rbp
  movq	%rsp, %rbp
  movl	$0x0, %eax
  popq	%rbp
  retq
  .cfi_endproc
.size foo.cold.1, .-foo.cold.1

  .local bar.cold.1
  .type bar.cold.1, %function
bar.cold.1:
# FDATA: 0 [unknown] 0 1 bar.cold.1/1 0 1 0
  .cfi_startproc
  pushq	%rbp
  movq	%rsp, %rbp
  movl	$0x0, %eax
  popq	%rbp
  retq
  .cfi_endproc
.size bar.cold.1, .-bar.cold.1
