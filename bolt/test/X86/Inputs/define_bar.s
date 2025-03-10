# Mocks a vtable object weak def in the C++ stdlib.
  .section .data.rel.ro,"aw"
  .weak bar
  .type bar, %object
bar:
  .space 20
  .size bar, 20
