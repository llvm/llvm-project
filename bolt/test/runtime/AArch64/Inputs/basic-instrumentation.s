  .globl main
  .type main, %function
main:
	sub		sp, sp, #16
	mov		w0, wzr
	str		wzr, [sp, #12]
	add		sp, sp, #16
	ret
.size main, .-main
