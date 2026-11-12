## If the operand references a symbol that differs from the jump table label,
## no reference updating is required even if its target address resides within
## the jump table's range.
## In this test case, consider the second instruction within the main function,
## where the address resulting from 'c + 17' corresponds to one byte beyond the
## address of the .LJTI2_0 jump table label. However, this operand represents
## an offset calculation related to the global variable 'c' and should remain
## unaffected by the jump table.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -no-pie %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt --funcs=main,foo/1 %t.exe -o %t.exe.bolt --print-normalized \
# RUN:   2>&1 | FileCheck %s

	.text
	.globl	main
	.type	main,@function
main:
# CHECK: Binary Function "main
	pushq   %rbp
	movq	%rsp, %rbp
	movq	$-16, %rax
	movl	c+17(%rax), %edx
# CHECK: movl	c+17(%rax), %edx
	cmpl	$255, %edx
	je	.LCorrect
	movl	$1, %eax
	popq	%rbp
	ret
.LCorrect:
	movl	$0, %eax
	popq	%rbp
	ret

	.p2align	4, 0x90
	.type	foo,@function
foo:
# CHECK: Binary Function "foo
	movq	$0, %rax
	jmpq	*.LJTI2_0(,%rax,8)
# CHECK: jmpq	*{{.*}} # JUMPTABLE
	addl	$-36, %eax
.LBB2_2:
	addl	$-16, %eax
	retq
	.section	.rodata,"a",@progbits
	.type	c,@object
	.data
	.globl	c
	.p2align	4, 0x0
c:
	.byte 1
  .byte 0xff
	.zero	14
	.size	c, 16
.LJTI2_0:
	.quad	.LBB2_2
	.quad	.LBB2_2
	.quad	.LBB2_2
	.quad	.LBB2_2

