.text

.globl  foo
.type foo, @function
foo:
  ret
  .size foo, .-foo

.globl  main
.type main, @function
main:
  .cfi_startproc

  cmp   %rdi, 0
.L1:
  jmp foo
  je .L1
  retq

  .cfi_endproc
  .size main, .-main
