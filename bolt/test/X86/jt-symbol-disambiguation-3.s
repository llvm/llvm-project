# In this test case, we reproduce the behavior seen in gcc where the
# base address of a jump table is decremented by some number and ends up
# at the exact addess of a jump table from another function. After
# linking, the instruction references another jump table and that
# confuses BOLT.
# We repro here the following issue:
# Before assembler: Instruction operand is: jumptable - 32
# After linking:    Instruction operand is: another_jumptable

# REQUIRES: system-linux

# XFAIL: *

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags -no-pie -nostartfiles -nostdlib -lc %t.o -o %t.exe -Wl,-q

# RUN: llvm-bolt %t.exe -o %t.exe.bolt --relocs=1 --lite=0 \
# RUN:   --reorder-blocks=reverse

# Useful when manually testing this. Currently we just check that
# the test does not cause BOLT to assert.
# COM: %t.exe.bolt 1 2

  .file "jt-symbol-disambiguation-3.s"
  .text

# ----
# Func foo contains a jump table whose start is colocated with a
# jump table reference in another function. However, the other function
# does not use the first entries of it and is merely doing arithmetics
# to save the creation of unused first entries.
# ----
  .globl foo
  .type foo, @function
foo:
  .cfi_startproc
  xor    %rax,%rax
  cmpq   $3, %rdi
  ja     .LBBAD
  jmpq   *.JT1(,%rdi,8)
.LBB1:
  movl   $0x4,%eax
  jmp    .LBB5
.LBB2:
  movl   $0x5,%eax
  jmp    .LBB5
.LBB3:
  movl   $0x6,%eax
  jmp    .LBB5
.LBB4:
  movl   $0x7,%eax
.LBB5:
  retq
.LBBAD:
  mov    $1, %rdi
  callq  exit@PLT
  retq
  .cfi_endproc
  .size foo, .-foo

# ----
# Func _start scans an object with indexed access using %rax * 8 as an
# index. However, %rax is known to be at least one, so the compiler
# loads the pointer for the base address as object - 8 instead of just
# object.
# ----
  .globl _start
  .type _start, @function
_start:
  .cfi_startproc
  movq   (%rsp), %rdi
  callq  foo
  movq   $1, %rdi
  cmpq   $7, %rax
  ja     .LBB10
  # Here the compiler uses the knowledge that the first four entries
  # of the jump table are not accessed and subtracts 32 from the base
  # address of it, so it doesn't have to allocate four unused entries in
  # memory. Unfortunately this can confuse BOLT since it ends up being a
  # direct reference to JT1, after linker is done.
  jmpq   *.JT2-32(,%rax,8)
.LBB6:
  xorq   %rdi, %rdi
  jmp .LBB10
.LBB7:
  xorq   %rdi, %rdi
  jmp .LBB10
.LBB8:
  xorq   %rdi, %rdi
  jmp .LBB10
.LBB9:
  xorq   %rdi, %rdi
  jmp .LBB10
.LBB10:
  callq exit@PLT
  retq
  .cfi_endproc
  .size _start, .-_start

# ----
# Data section
# ----
  .section .rodata,"a",@progbits
  .p2align 3
.JT1:
  .quad .LBB1
  .quad .LBB2
  .quad .LBB3
  .quad .LBB4
.JT2:
  .quad .LBB6
  .quad .LBB7
  .quad .LBB8
  .quad .LBB9
