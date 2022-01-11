# This reproduces a bug with shrink wrapping when trying to move
# push-pops in a function where we are not allowed to modify the
# stack layout for alignment reasons. In this bug, we failed to
# propagate alignment requirement upwards in the call graph for
# some functions when there is a cycle in the call graph.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# Delete our BB symbols so BOLT doesn't mark them as entry points
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q

# RUN: llvm-bolt %t.exe -relocs=1 -frame-opt=all -print-finalized \
# RUN:   -lite=0 -print-only=main -data %t.fdata -o %t.out | FileCheck %s

# RUN: %t.out

# CHECK: BOLT-INFO: Shrink wrapping moved 1 spills inserting load/stores and 0 spills inserting push/pops

  .text
  .globl bar
  .type bar, %function
  .p2align 4
bar:
# FDATA: 0 [unknown] 0 1 bar 0 0 510
  pushq %rbp
  movq  %rsp, %rbp
  pushq %rbx                  # We save rbx here, but there is an
                              # opportunity to move it to .BB2
  subq  $0x18, %rsp
  cmpl  $0x2, %edi
.J1:
  jb    .BBend
# FDATA: 1 bar #.J1# 1 bar #.BB2# 0 10
# FDATA: 1 bar #.J1# 1 bar #.BBend# 0 500
.BB2:
  movq $2, %rbx               # Use rbx in a cold block
  xorq %rax, %rax
  movb mystring, %al
  addq %rbx, %rax
  movb %al, mystring
  leaq mystring, %rdi
  #callq puts

.BBend:
  addq $0x18, %rsp
  pop %rbx                    # Restore
  xorq %rax, %rax
  cmpq  $0x0, %rax
  jne  .BBnever
  jmp  .BBbarend
.BBnever:
  # This is a path that is never executed, but we add a call to main here
  # to force a cycle in the call graph and to require us to have an aligned
  # stack
  callq main
.BBbarend:
  leaveq
  retq
  .size bar, .-bar

# Frame alignedness information needs to be transmitted from foo to main to bar
  .globl  main
  .type main, %function
  .p2align  4
main:
  # Call a function that requires an aligned stack
  callq foo
  # Call a function that can be shrink-wrapped
  callq bar
  retq
  .size main, .-main

# Frame alignedness information needs to be transmitted from foo to main to bar
  .globl  foo
  .type foo, %function
  .p2align  4
foo:
  # Use an instruction that requires an aligned stack
  movdqa -0x10(%rsp), %xmm0
  retq
  .size foo, .-foo

  .data
mystring: .asciz "0 is rbx mod 10 contents in decimal\n"
