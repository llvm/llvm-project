# In this test case, the symbol that represents the end of a table
# in .rodata is being colocated with the start of a jump table from
# another function, and BOLT moves that jump table. This should not
# cause the symbol representing the end of the table to be moved as
# well.
# Bug reported in https://github.com/llvm/llvm-project/issues/55004

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags -no-pie -nostartfiles -nostdlib -lc %t.o -o %t.exe -Wl,-q

# RUN: llvm-bolt %t.exe -o %t.exe.bolt --relocs=1 --lite=0 \
# RUN:   --reorder-blocks=reverse -jump-tables=move

# RUN: %t.exe.bolt 1 2 3

  .file "jt-symbol-disambiguation.s"
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
# Func _start scans a table using begin/end pointers. End pointer is colocated
# with the start of a jump table of function foo. When that jump
# table moves, end pointer in _start should not be affected.
# ----
  .globl _start
  .type _start, @function
_start:
  .cfi_startproc
  movq   (%rsp), %rdi
  callq foo
  leaq   .start_of_table(%rip), %rsi  # iterator
  leaq   .end_of_table(%rip), %rdi    # iterator end
.LBB6:
  cmpq %rsi, %rdi
  je .LBB7
  movq (%rsi), %rbx
  leaq 8(%rsi), %rsi            # ++iterator
  jmp .LBB6
.LBB7:
  xor   %rdi, %rdi
  callq exit@PLT
  .cfi_endproc
  .size _start, .-_start

# ----
# Data section
# ----
  .section .rodata,"a",@progbits
  .p2align 3
.start_of_table:
  .quad 123
  .quad 456
  .quad 789
.end_of_table:
.JT1:
  .long .LBB1 - .JT1
  .long .LBB2 - .JT1
  .long .LBB3 - .JT1
  .long .LBB4 - .JT1
