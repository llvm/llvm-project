.text

.globl	foo
.type	foo, @function
foo:
  ret
	.size	foo, .-foo

.globl	main
.type	main, @function
main:
	.cfi_startproc

  movsd   8(%rdi), %xmm0
  ucomisd 8(%rsi), %xmm0
  jp      .Lend
  jne     .Lend
  jmp     foo
.Lend:
  xorl    %eax, %eax
  retq

	.cfi_endproc
	.size	main, .-main

