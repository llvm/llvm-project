## This reproduces issue 20 from our github repo
## "BOLT crashes when removing unreachable BBs that are a target
##   in a JT"

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags -no-pie %t.o -o %t.exe
# RUN: llvm-bolt %t.exe --relocs=0 --jump-tables=move --print-finalized \
# RUN:    -o %t.out | FileCheck %s

# CHECK-NOT: BOLT-INFO: UCE removed {{.*}} blocks and {{.*}} bytes of code
# CHECK: Binary Function "main"
# CHECK:      .LFT{{.*}} (2 instructions, align : 1)
# CHECK-NEXT:     00000004:   andq
# CHECK-NEXT:     00000008:   jmpq
# CHECK-NEXT:   Successors: .Ltmp{{.*}}, .Ltmp{{.*}}, .Ltmp{{.*}}, .Ltmp{{.*}}


  .text
  .globl main
  .type main, %function
  .size main, .Lend1-main
main:
  xorq %rax, %rax
  retq
  andq $3, %rdi
  jmpq *jumptbl(,%rdi,8)

.Lbb1:
  movl $0x1, %eax
  jmp .Lexit
.Lbb2:
  movl $0x2, %eax
  jmp .Lexit
.Lbb3:
  movl $0x3, %eax
  jmp .Lexit
.Lbb4:
  movl $0x4, %eax
.Lexit:
  retq
.Lend1:

  .section .rodata
  .globl jumptbl
jumptbl:
  .quad .Lbb1
  .quad .Lbb2
  .quad .Lbb3
  .quad .Lbb4
