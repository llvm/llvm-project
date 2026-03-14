.text

.globl	foo
.type	foo, @function
foo:
  cmp $1, %rdi
  je bar
  jmp .L2
.L1:
  ret
.L2:
	cmpq $1, %rdi
  jne .L1
	.size	foo, .-foo

.globl  bar
.type	bar, @function
bar:
  ret
	.size	bar, .-bar

.globl	main
.type	main, @function
main:
.LFB1:
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
.LFE1:
	.size	main, .-main
