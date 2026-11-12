  .globl main
  .type main, %function
main:
	pushq	%rax
	callq	foo
	movl	$0x1, %eax
	popq	%rdx
	retq
.size main, .-main

  .globl foo
  .type foo, %function
foo:
	jmp	bar
.size foo, .-foo

  .globl bar
  .type bar, %function
bar:
	retq
.size bar, .-bar
