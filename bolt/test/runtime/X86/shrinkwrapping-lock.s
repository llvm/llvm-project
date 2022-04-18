# This reproduces a bug with shrink wrapping when moving
# load instructions in-between the lock prefix and another
# instruction (where the lock prefix applies).

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# Delete our BB symbols so BOLT doesn't mark them as entry points
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags -no-pie %t.o -o %t.exe -Wl,-q

# RUN: llvm-bolt %t.exe -relocs=1 -frame-opt=all -print-fop \
# RUN:    -print-only=main -data %t.fdata -o %t.out | FileCheck %s

# RUN: %t.out 1

# CHECK: BOLT-INFO: Shrink wrapping moved 1 spills inserting load/stores and 0 spills inserting push/pops

  .text
  .globl  main
  .type main, %function
  .p2align  4
main:
# FDATA: 0 [unknown] 0 1 main 0 0 510
  pushq %rbp
  movq  %rsp, %rbp
  pushq %rbx                  # We save rbx here, but there is an
                              # opportunity to move it to .LBB2
  subq  $0x18, %rsp
  cmpl  $0x2, %edi
.J1:
  jb    .BBend
# FDATA: 1 main #.J1# 1 main #.BB2# 0 10
# FDATA: 1 main #.J1# 1 main #.BBend# 0 500
.BB2:
  movq $2, %rbx               # Use rbx in a cold block. Save rbx will be moved
                              # just before this instruction.
  xorq %rax, %rax
  movb mystring, %al
  addq %rbx, %rax
  movb %al, mystring
  leaq mystring, %rdi
  callq puts
  lock add %r12,0x0(%rsp)     # Put a lock in an unrelated instruction at the
                              # dom. frontier where the restore will be moved to
                              # We should not put the restore in-between the
                              # lock and add! We typically avoid putting a
                              # restore in the last BB instruction, but since
                              # lock in llvm MC lib is considered a
                              # separate instruction, we may mistakenly
                              # put the restore just between these two.

.BBend:
  addq $0x18, %rsp
  pop %rbx                    # Restore should be moved
  xorq %rax, %rax
  leaveq
  retq
  .size main, .-main

  .data
mystring: .asciz "0 is rbx mod 10 contents in decimal\n"
