## This checks that shrink wrapping does not pessimize a CFG pattern where two
## blocks can be proved to have the same execution count but, because of profile
## inaccuricies, we could move saves into the second block. We can prove two
## blocks have the same frequency when B post-dominate A and A dominates B and
## are at the same loop nesting level. This would be a pessimization because
## shrink wrapping is unlikely to be able to cleanly move PUSH instructions,
## inserting additional store instructions.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt -relocs %t.exe -o %t.out -data %t.fdata \
# RUN:     -frame-opt=all -equalize-bb-counts | FileCheck %s

## Here we create a CFG pattern with two blocks A and B belonging to the same
## equivalency class as defined by dominance relations and having in theory
## the same frequency. But we tweak edge counts from profile to make block A
## hotter than block B.
  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
## Hot prologue
# FDATA: 0 [unknown] 0 1 _start 0 0 10
  push  %rbp
  mov   %rsp, %rbp
  push  %rbx
  push  %r14
  subq  $0x20, %rsp
b:  je  end_if_1
# FDATA: 1 _start #b# 1 _start #end_if_1# 0 1
if_false:
  movq rel(%rip), %rdi  # Add this to create a relocation and run bolt w/ relocs
c:  jmp end_if_1
## Reduce frequency from 9 to 1 to simulate an inaccurate profile
# FDATA: 1 _start #c# 1 _start #end_if_1# 0 1
end_if_1:
  # first uses of R14 and RBX appear at this point, possible move point for SW
  mov %r14, %rdi
  mov %rbx, %rdi
  leaq -0x20(%rbp), %r14
  movq -0x20(%rbp), %rdi
  addq  $0x20, %rsp
  pop %r14
  pop %rbx
  pop %rbp
  ret
  .cfi_endproc
  .size _start, .-_start

  .data
rel:  .quad _start

# CHECK:   BOLT-INFO: Shrink wrapping moved 0 spills inserting load/stores and 0 spills inserting push/pops
