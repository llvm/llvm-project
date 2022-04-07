# This reproduces a bug with BOLT non-reloc mode, during emission, if the user
# does not use -update-debug-sections. In this bug, if a function gets too large
# to occupy its original location, but it has a jump table, BOLT would skip
# rewriting the function but it would still overwrite the jump table in a bogus
# file offset (offset zero). This will typically corrupt the .interp section,
# which is the first section in the binary, depending on the size of the jump
# table that was written. If .interp is corrupted, the binary won't run.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags -no-pie -nostartfiles -nostdlib -lc %t.o -o %t.exe

# RUN: llvm-bolt %t.exe -o %t.exe.bolt -relocs=0 -lite=0 -reorder-blocks=reverse

# RUN: %t.exe.bolt 1 2 3

  .file "test.S"
  .text
  .globl _start
  .type _start, @function
_start:
  .cfi_startproc
  xor    %rax,%rax
  movq   (%rsp), %rdi
  and    $0x3,%rdi
  jmpq   *.JT1(,%rdi,8)
.LBB1:
  movl   $0x1,%eax
  jmp    .LBB5
.LBB2:
  movl   $0x2,%eax
  jmp    .LBB5
.LBB3:
  movl   $0x3,%eax
  jmp    .LBB5
.LBB4:
  movl   $0x4,%eax
.LBB5:
  callq exit@PLT
  .cfi_endproc
  .size _start, .-_start

# Make the jump table large enough to force the bug to manifest as .interp
# being corrupt. Typically .interp will be at offset 0x1c8, so the jump table
# needs to be larger than that.
  .section .rodata,"a",@progbits
  .p2align 3
.JT1:
  .quad .LBB1
  .quad .LBB2
  .quad .LBB3
  .quad .LBB4
  .quad .LBB5
  .quad .LBB5
  .rept 100
  .quad .LBB1
  .endr
