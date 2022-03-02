## Check that BOLT correctly recognizes pc-relative function pointer
## reference with an addend.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux %s -o %t.o
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: ld.lld %t.o -o %t.exe -q
# RUN: llvm-bolt %t.exe -relocs -o /dev/null -print-only=foo -print-disasm \
# RUN:   | FileCheck %s

  .text
  .globl _start
  .type _start,@function
_start:
  .cfi_startproc
  call foo
  retq
  .size _start, .-_start
  .cfi_endproc

  .globl  foo
  .type foo,@function
foo:
  .cfi_startproc

  leaq  foo-1(%rip), %rax
## Check that the instruction references foo with a negative addend,
## not the previous function with a positive addend (_start+X).
#
# CHECK: leaq    foo-1(%rip), %rax

  retq
  .size foo, .-foo
  .cfi_endproc
