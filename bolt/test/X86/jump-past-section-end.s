## Check that a jump past the end of a function that is the last one in its
## section (result of __builtin_unreachable()) is treated as a nop instead of
## an interprocedural reference to an address outside of any section.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.out --lite=0 --print-cfg \
# RUN:   --print-only=main,foo 2>&1 | FileCheck %s

# CHECK: BOLT-WARNING: jump past end detected at 0x{{.*}} in function foo

  .section first_text,"ax",@progbits
  .globl main
  .type main, %function
main:
  call foo
  xorl %eax, %eax
  retq
  .size main, .-main

  .section last_text,"ax",@progbits
  .p2align 6
  .globl foo
  .type foo, %function
foo:
  call abort
  jmp .Lend
.Lend:
  .size foo, .-foo

## Align creates an empty space between the end of foo and the next section,
## to repro the bug where end of foo is truly outside of any section.
  .section next_text,"ax",@progbits
  .p2align 6
  .byte 0
