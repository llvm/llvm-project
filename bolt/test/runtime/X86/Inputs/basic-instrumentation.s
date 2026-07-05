  .globl main
  .type main, %function
main:
	pushq	%rbp
	movq	%rsp, %rbp
	movl	$0x0, %eax
	popq	%rbp
	retq
.size main, .-main
