.file	"ft_to_noop.s"
.text

.globl	foo
.type	foo, @function
foo:
LFB0:
# FDATA: 0 [unknown] 0 1 foo 0 0 20
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	$0, -4(%rbp)
	cmpl	$10, -20(%rbp)
LBB00_br:
	jle	L2
  nop
# FDATA: 1 foo #LBB00_br# 1 foo #L2# 0 18
# FDATA: 1 foo #LBB00_br# 1 foo #LFT0# 0 3

LFT0:
	movl	-20(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -4(%rbp)
L2:
	addl	$1, -4(%rbp)
	movl	-4(%rbp), %eax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
LFE0:
	.size	foo, .-foo

.globl	main
.type	main, @function
main:
LFB1:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movl	%edi, -4(%rbp)
	movq	%rsi, -16(%rbp)
	movl	-4(%rbp), %eax
	movl	%eax, %edi
	call	foo
	movl	$0, %eax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
LFE1:
	.size	main, .-main
