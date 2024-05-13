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

  cmp   %rdi, 1
  jne  .L2

  xorl    %eax, %eax

.L1:
  jmp foo

.L2:
  jb .L1

  cmp %eax, 0
  xorl    %eax, %eax
  ja .L2
.Lend:
  retq

	.cfi_endproc
	.size	main, .-main

