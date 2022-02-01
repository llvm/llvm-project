## Tests the peephole that adds trap instructions following indirect tail calls.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt -peepholes=tailcall-traps \
# RUN:   -print-peepholes -funcs=foo,bar 2>&1 | FileCheck %s

# CHECK:  Binary Function "foo"
# CHECK:        br     x0  # TAILCALL
# CHECK-NEXT:   brk    #0x1
# CHECK:  End of Function "foo"

# CHECK:  Binary Function "bar"
# CHECK:        b     foo # TAILCALL
# CHECK:  End of Function "bar"

  .text
  .align 4
  .global main
  .type main, %function
main:
  nop
  ret
  .size main, .-main

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
