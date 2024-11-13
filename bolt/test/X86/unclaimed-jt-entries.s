## This test ensures that "unclaimed" jump table entries are accounted later
## in postProcessIndirectBranches and the function is marked as non-simple.

# The test is compiled from the following source using GCC 12.2 -O3:
# https://godbolt.org/z/YcPG131s6
# int func(long long Input) {
#   switch(Input) {
#   case 3: return 1;
#   case 4: return 2;
#   case 6: return 3;
#   case 8: return 4;
#   case 13: return 5;
#   default: __builtin_unreachable();
#   }
# }

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -no-pie %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -v=1 -o %t.out 2>&1 | FileCheck %s

# CHECK: BOLT-WARNING: unclaimed data to code reference (possibly an unrecognized jump table entry) to .Ltmp[[#]] in main
# CHECK: BOLT-WARNING: unclaimed data to code reference (possibly an unrecognized jump table entry) to .Ltmp[[#]] in main
# CHECK: BOLT-WARNING: unclaimed data to code reference (possibly an unrecognized jump table entry) to .Ltmp[[#]] in main
# CHECK: BOLT-WARNING: unclaimed data to code reference (possibly an unrecognized jump table entry) to .Ltmp[[#]] in main
# CHECK: BOLT-WARNING: unclaimed data to code reference (possibly an unrecognized jump table entry) to .Ltmp[[#]] in main
# CHECK: BOLT-WARNING: failed to post-process indirect branches for main

  .text
  .globl main
  .type main, %function
  .size main, .Lend-main
main:
  jmp *L4-24(,%rdi,8)
.L5:
  movl $4, %eax
  ret
.L9:
  movl $2, %eax
  ret
.L8:
  movl $1, %eax
  ret
.L3:
  movl $5, %eax
  ret
.L6:
  movl $3, %eax
  ret
.Lend:

.section .rodata
  .globl L4
L4:
  .quad .L8
  .quad .L9
  .quad .L3
  .quad .L6
  .quad .L3
  .quad .L5
  .quad .L3
  .quad .L3
  .quad .L3
  .quad .L3
  .quad .L3
