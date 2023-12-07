# Check that functions listed in -function-order list take precedence over
# lite mode function filtering.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe --data %t.fdata --lite --reorder-functions=user \
# RUN:   --function-order=%p/Inputs/order-lite.txt -o %t -print-all 2>&1 \
# RUN:   | FileCheck %s

# CHECK: 1 out of 3 functions in the binary (33.3%) have non-empty execution profile
# CHECK-DAG: Binary Function "main" after reorder-functions
# CHECK-DAG: Binary Function "foo.__uniq.123" after reorder-functions

  .globl main
  .type main, %function
main:
	.cfi_startproc
.LBB06:
	callq	testfunc
	retq
	.cfi_endproc
.size main, .-main

  .globl testfunc
  .type testfunc, %function
testfunc:
# FDATA: 0 [unknown] 0 1 testfunc 0 1 0
	.cfi_startproc
	pushq	%rbp
	movq	%rsp, %rbp
	movl	$0x0, %eax
	popq	%rbp
	retq
	.cfi_endproc
.size testfunc, .-testfunc

  .globl foo.__uniq.123
  .type foo.__uniq.123, %function
foo.__uniq.123:
	.cfi_startproc
	pushq	%rbp
	movq	%rsp, %rbp
	movl	$0x0, %eax
	popq	%rbp
	retq
	.cfi_endproc
.size foo.__uniq.123, .-foo.__uniq.123
