# Test case with a simple PIC-style jump table where the register containing
# the jump table address is defined outside of the basic block containing the
# jump on register.
#
# One of the destinations of the jump table points past the end of the function
# similar to the code generated for __builtin_unreachable().

	.text
	.globl	main
	.type	main, @function
main:
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	leaq	.LJUMPTABLE(%rip), %rcx
.L13:
	cmpl	$3, %edi
	ja	.L2

	movslq	(%rcx,%rdi,4), %rax
	addq	%rcx, %rax
	jmp	*%rax

.L12:
	movq	%rdi, %rax
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret

.L5:
	.cfi_restore_state
	addq	$9, %rdi
	jmp	.L12
.L6:
	addq	$8, %rdi
	jmp	.L12
.L2:
	addq	$2, %rdi
	jmp	.L12
.LUNREACHABLE:
	.cfi_endproc
	.size	main, .-main

	.section	.rodata
	.align 4
.LJUMPTABLE:
	.long	.L2-.LJUMPTABLE
	.long	.L6-.LJUMPTABLE
	.long	.L5-.LJUMPTABLE
	.long	.LUNREACHABLE-.LJUMPTABLE
