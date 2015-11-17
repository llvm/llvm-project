  .text
  .globl hey;
  .type hey, @function;
hey:
  .type hey, @gnu_indirect_function;
  leaq  __hey_1(%rip), %rax
  ret

  .text
  .type __hey_1, @function;
  .globl __hey_1;
__hey_1:
  movq $42, %rax
  ret

  .text
  .type plt, @function;
  .globl plt;
plt:
  call hey@PLT
  ret
