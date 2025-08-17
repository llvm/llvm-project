  .globl main
  .type main, %function
main:
  .cfi_startproc
  cmpq $0x3, %rdi
  jae .L4
  cmpq $0x1, %rdi
  jne .L4
  movslq .Ljt_pic+8(%rip), %rax
  lea .Ljt_pic(%rip), %rdx
  add %rdx, %rax
  jmpq *%rax
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
.Ljt_pic:
  .long .L1 - .Ljt_pic
  .long .L2 - .Ljt_pic
  .long .L3 - .Ljt_pic
  .long .L4 - .Ljt_pic

