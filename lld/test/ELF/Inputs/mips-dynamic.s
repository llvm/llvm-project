  .option pic2
  .text
  .globl _foo
_foo:
  nop

  .globl foo
  .type foo, @function
foo:
  nop

  .data
  .globl data0
  .type data0, @object
data0:
  .word 0

  .globl data1
  .type data1, @object
data1:
  .word 0
