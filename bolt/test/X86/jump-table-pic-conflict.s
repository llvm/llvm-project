# Check cases when the first PIC jump table entries of one function can be
# interpreted as valid last entries of the previous function.

# Conditions to trigger the bug:  Function A and B have jump tables that
# are adjacent in memory. We run in lite relocation mode. Function B
# is not disassembled because it does not have profile. Function A
# triggers a special conditional that forced BOLT to rewrite its jump
# table in-place (instead of moving it) because it is marked as
# non-simple (in this case, containing unknown control flow). The
# first entry of B's jump table (a PIC offset) happens to be a valid
# address inside A when added to A's jump table base address. In this
# case, BOLT could overwrite B's jump table, corrupting it, thinking
# the first entry of it is actually part of A's jump table.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: ld.lld %t.o -o %t.exe -q -T %S/Inputs/jt-pic-linkerscript.ld
# RUN: llvm-bolt %t.exe -relocs -o %t.out -data %t.fdata \
# RUN:     -lite=1
# RUN: llvm-readelf -S %t.out | FileCheck --check-prefix=CHECK %s
# The output binary is runnable, but we check for test success with
# readelf. This is another way to check this bug:
# COM: %t.out

# BOLT needs to create a new rodata section, indicating that it
# successfully moved the jump table in _start.
# CHECK: [{{.*}}] .bolt.org.rodata

  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
# FDATA: 0 [unknown] 0 1 _start 0 0 1
  push    %rbp
  mov     %rsp, %rbp
  mov     0x8(%rbp), %rdi
  cmpq    $3, %rdi
  ja      .L5
  jmp     .L6
# Unreachable code, here to mark this function as non-simple
# (containing unknown control flow) with a stray indirect jmp
  jmp     *%rax
.L6:
  decq    %rdi
  leaq    .LJT1(%rip), %rcx
  movslq  (%rcx, %rdi, 4), %rax
  addq    %rcx, %rax
  jmp     *%rax
.L1:
  leaq    str1(%rip), %rsi
  jmp     .L4
.L2:
  leaq    str2(%rip), %rsi
  jmp     .L4
.L3:
  leaq    str3(%rip), %rsi
  jmp     .L4
.L5:
  leaq    str4(%rip), %rsi
.L4:
  movq    $1, %rdi
  movq    $10, %rdx
  movq    $1, %rax
  syscall
  mov     0x8(%rbp), %rdi
  decq    %rdi
  callq   func_b
  movq    %rax, %rdi
  movq    $231, %rax
  syscall
  pop     %rbp
  ret
  .cfi_endproc
  .size _start, .-_start

  .globl func_b
  .type func_b, %function
func_b:
  .cfi_startproc
  push    %rbp
  mov     %rsp, %rbp
  cmpq    $3, %rdi
  ja      .L2_6
# FT
  leaq    .LJT2(%rip), %rcx
  movslq  (%rcx, %rdi, 4), %rax
  addq    %rcx, %rax
  jmp     *%rax
.L2_1:
  movq    $0, %rax
  jmp     .L2_5
.L2_2:
  movq    $1, %rax
  jmp     .L2_5
.L2_3:
  movq    $2, %rax
  jmp     .L2_5
.L2_4:
  movq    $3, %rax
  jmp     .L2_5
.L2_6:
  movq    $-1, %rax
.L2_5:
  popq    %rbp
  ret
  .cfi_endproc
  .size func_b, .-func_b

  .rodata
str1: .asciz "Message 1\n"
str2: .asciz "Message 2\n"
str3: .asciz "Message 3\n"
str4: .asciz "Highrange\n"
# Special case where the first .LJT2 entry is a valid offset of
# _start when interpreted with .LJT1 as a base address.
.LJT1:
  .long .L1-.LJT1
  .long .L2-.LJT1
  .long .L3-.LJT1
  .long .L3-.LJT1
  .long .L3-.LJT1
  .long .L3-.LJT1
  .long .L3-.LJT1
.LJT2:
  .long .L2_1-.LJT2
  .long .L2_2-.LJT2
  .long .L2_3-.LJT2
  .long .L2_4-.LJT2
