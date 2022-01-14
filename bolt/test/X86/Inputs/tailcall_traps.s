.globl main
main:

.globl foo
foo:
	.cfi_startproc
  jmpq	*%rax
	.cfi_endproc

.globl bar
bar:
	.cfi_startproc
  jmp	moo
	.cfi_endproc

.globl moo
moo:
  nop
