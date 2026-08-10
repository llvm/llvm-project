## Tests the peephole that adds trap instructions following indirect tail calls.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --peepholes=tailcall-traps \
# RUN:   --print-peepholes --funcs=foo,bar 2>&1 | FileCheck %s

# CHECK:  Binary Function "foo"
# CHECK:        br     x0  # TAILCALL
# CHECK-NEXT:   brk    #0x1
# CHECK:  End of Function "foo"

# CHECK:  Binary Function "bar"
# CHECK:        b     foo # TAILCALL
# CHECK:  End of Function "bar"

  .text
  .align 4
  .global _start
  .type _start, %function
_start:
  nop
  ret
  .size _start, .-_start

  .global foo
  .type foo, %function
foo:
  br x0
  .size foo, .-foo

  .global bar
  .type bar, %function
bar:
  b foo
  .size bar, .-bar
