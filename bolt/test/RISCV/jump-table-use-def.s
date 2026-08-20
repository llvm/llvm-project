// REQUIRES: system-linux,target=riscv64{{.*}}

// Verify that RISC-V jump-table recognition follows the local register
// use-def chain instead of relying on adjacent instructions. Exercise both
// GCC-style absolute 32-bit entries and PIC-relative 32-bit entries.

// RUN: %clang %cflags64 -march=rv64gc_zba -no-pie \
// RUN:   -Wl,--no-relax,--image-base=0x10000,--section-start=.text=0x20000,--section-start=.rodata=0x30000 \
// RUN:   -o %t %s
// RUN: %t
// RUN: llvm-bolt %t -o %t.bolt --jump-tables=move --print-cfg \
// RUN:   --print-jump-tables --print-only=abs_dispatch,pic_dispatch 2>&1 | \
// RUN:   FileCheck %s
// RUN: %t.bolt

// RUN: %clang %cflags32 -march=rv32imac_zba -no-pie \
// RUN:   -Wl,--no-relax,--image-base=0x10000,--section-start=.text=0x20000,--section-start=.rodata=0x30000 \
// RUN:   -o %t.rv32 %s
// RUN: llvm-bolt %t.rv32 -o %t.rv32.bolt --jump-tables=move --print-cfg \
// RUN:   --print-jump-tables --print-only=abs_dispatch,pic_dispatch 2>&1 | \
// RUN:   FileCheck %s

// CHECK-LABEL: Binary Function "abs_dispatch"
// CHECK: jr a1 # JUMPTABLE @0x30000
// CHECK-LABEL: Binary Function "pic_dispatch"
// CHECK: jr a1 # JUMPTABLE @0x3000c
// CHECK: Jump table ABS_JT for function abs_dispatch
// CHECK: PIC Jump table PIC_JT for function pic_dispatch

  .text
  .globl _start
  .type _start, @function
  .p2align 2
_start:
  li a0, 1
  call abs_dispatch
  li t0, 11
  bne a0, t0, .Lfail

  li a0, 2
  call pic_dispatch
  li t0, 22
  bne a0, t0, .Lfail

  li a0, 0
  li a7, 93
  ecall
.Lfail:
  li a0, 1
  li a7, 93
  ecall
  .size _start, .-_start

  .globl abs_dispatch
  .type abs_dispatch, @function
  .p2align 2
abs_dispatch:
  lui a4, %hi(ABS_JT)
  addi a4, a4, %lo(ABS_JT)
  li t0, 7                         // Unrelated instruction in the def chain.
  sh2add a1, a0, a4
  li t1, 8                         // Unrelated instruction in the def chain.
  lw a1, 0(a1)
  li t2, 9                         // The load need not be adjacent to JR.
  jr a1
.Labs0:
  li a0, 10
  ret
.Labs1:
  li a0, 11
  ret
.Labs2:
  li a0, 12
  ret
  .size abs_dispatch, .-abs_dispatch

  .globl pic_dispatch
  .type pic_dispatch, @function
  .p2align 2
pic_dispatch:
.Lpcrel_hi:
  auipc a4, %pcrel_hi(PIC_JT)
  addi a4, a4, %pcrel_lo(.Lpcrel_hi)
  li t0, 7                         // Unrelated instruction in the def chain.
  slli a1, a0, 2
  add a1, a1, a4
  li t1, 8                         // Unrelated instruction in the def chain.
  lw a1, 0(a1)
  li t2, 9                         // Separate the load, ADD, and JR.
  add a1, a1, a4
  jr a1
.Lpic0:
  li a0, 20
  ret
.Lpic1:
  li a0, 21
  ret
.Lpic2:
  li a0, 22
  ret
  .size pic_dispatch, .-pic_dispatch

  .section .rodata,"a",@progbits
  .globl ABS_JT
  .type ABS_JT, @object
  .p2align 2
ABS_JT:
  .word .Labs0
  .word .Labs1
  .word .Labs2
  .size ABS_JT, .-ABS_JT

  .globl PIC_JT
  .type PIC_JT, @object
  .p2align 2
PIC_JT:
  .word .Lpic0 - PIC_JT
  .word .Lpic1 - PIC_JT
  .word .Lpic2 - PIC_JT
  .size PIC_JT, .-PIC_JT
