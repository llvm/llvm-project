  .globl main
  .type main, %function
main:
	subq	$0x8, %rsp
	callq	foo
	movl	$0x400638, %edi
	callq	puts@PLT
	xorl	%eax, %eax
	addq	$0x8, %rsp
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
	jmp	baz
.size bar, .-bar

  .globl baz
  .type baz, %function
baz:
	retq
.size baz, .-baz
