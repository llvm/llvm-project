.globl foo
foo:
	.cfi_startproc
  jmpq	*%rax
	.cfi_endproc
.size foo, .-foo

.globl bar
bar:
	.cfi_startproc
  jmp	moo
	.cfi_endproc
.size bar, .-bar

.globl moo
moo:
  nop
.size moo, .-moo
