# In this test case, we reproduce the behavior seen in gcc where the
# base address of a data object is decremented by some number and lands
# inside a jump table from another function.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags -no-pie -nostartfiles -nostdlib -lc %t.o -o %t.exe -Wl,-q

# RUN: llvm-bolt %t.exe -o %t.exe.bolt --relocs=1 --lite=0 \
# RUN:   --reorder-blocks=reverse -jump-tables=move

# RUN: %t.exe.bolt 1 2 3

  .file "jt-symbol-disambiguation-2.s"
  .text

# ----
# Func foo contains a jump table whose start is colocated with a
# symbol marking the end of a data table
# ----
  .globl foo
  .type foo, @function
foo:
  .cfi_startproc
  xor    %rax,%rax
  and    $0x3,%rdi
  leaq   .JT1(%rip), %rax
  movslq  (%rax, %rdi, 4), %rdi
  addq   %rax, %rdi
  jmpq   *%rdi
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
  callq foo
  xorq   %rbx, %rbx
  leaq   .object-8(%rip), %rsi  # indexed access base address
  movq   $1, %rax               # start index
.LBB6:
  cmpq $4, %rax
  je .LBB7
  addq (%rsi,%rax,8), %rbx
  incq %rax   # ++iterator
  jmp .LBB6
.LBB7:
  cmpq  $1368, %rbx             # check .object contents integrity
  jne   .LBB_BAD
  xor   %rdi, %rdi
  callq exit@PLT
  retq
.LBB_BAD:
  leaq  .message, %rdi
  callq puts@PLT
  movq  $1, %rdi
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
  .long .LBB1 - .JT1
  .long .LBB2 - .JT1
  .long .LBB3 - .JT1
  .long .LBB4 - .JT1
.object:
  .quad 123
  .quad 456
  .quad 789
.message:
  .asciz "RUNTIME ASSERTION FAILURE: references in test binary are corrupt after BOLT"
