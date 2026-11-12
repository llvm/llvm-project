
  .text
  .globl main
  .type main, %function
main:
  .cfi_startproc
  cmpq $0x3, %rdi
  jae .L4
  cmpq $0x1, %rdi
  jne .Ldo_jump
  jmpq *jt+8
.Ldo_jump:
  jmpq *jt(,%rdi,8)
.L1:
  movq $0x1, %rax
  jmp .L5
.L2:
  movq $0x0, %rax
  jmp .L5
.L3:
  movq $0x2, %rax
  jmp .L5
.L4:
  mov $0x3, %rax
.L5:
  retq
  .cfi_endproc

  .section .rodata
  .align 16
  .globl jt
jt:
	.quad	.L1
	.quad	.L2
	.quad	.L3
