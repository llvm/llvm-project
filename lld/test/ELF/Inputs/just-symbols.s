.globl foo, bar
foo:
  ret

.section .data
.type bar, @object
.size bar, 40
bar:
  .zero 40
