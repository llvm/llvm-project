.type _a, @function
_a:
  .cfi_startproc
  test  %edx, %edx
  .cfi_endproc

.globl _b
.type _b, @function
_b:
  .cfi_startproc
  jnz   a
  .cfi_endproc

.globl _f
.type _f, @function
_f:
  .cfi_startproc
  mov   %rdx, %rcx
a:
  mov   0, %r9

.globl _e
.type _e, @function
_e:

.globl _d
.type _d, @function
_d:

.globl _c
.type _c, @function
_c:
  movdqu 0, %xmm3
  .cfi_endproc

.globl	main
.type	main, @function
main:
  nop

.globl _start
.type _start, @function
_start:
  jmp main
