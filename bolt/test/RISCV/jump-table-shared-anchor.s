// REQUIRES: system-linux,target=riscv64{{.*}}

// Do not treat RV64 full-width label-address arrays as movable jump tables.
// GCC can place multiple arrays at offsets from one shared anchor. Moving the
// array at offset zero changes the anchor while an unrecognized reference to a
// later array still relies on the original layout.

// RUN: %clang %cflags64 -march=rv64gc -no-pie \
// RUN:   -Wl,--no-relax,--image-base=0x10000,--section-start=.text=0x20000,--section-start=.rodata=0x30000 \
// RUN:   -o %t %s
// RUN: %t
// RUN: llvm-bolt %t -o %t.bolt --jump-tables=move --print-cfg \
// RUN:   --print-only=shared_first,shared_second 2>&1 | FileCheck %s
// RUN: %t.bolt

// CHECK-LABEL: Binary Function "shared_first"
// CHECK-NOT: JUMPTABLE
// CHECK: jr a1 # UNKNOWN CONTROL FLOW
// CHECK-LABEL: Binary Function "shared_second"
// CHECK-NOT: JUMPTABLE
// CHECK: jr a1 # UNKNOWN CONTROL FLOW

  .text
  .globl _start
  .type _start, @function
  .p2align 2
_start:
  li a0, 1
  call shared_second
  li t0, 41
  bne a0, t0, .Lfail

  li a0, 0
  li a7, 93
  ecall
.Lfail:
  li a0, 1
  li a7, 93
  ecall
  .size _start, .-_start

  .globl shared_first
  .type shared_first, @function
  .p2align 2
shared_first:
  lui a4, %hi(SHARED_ANCHOR)
  addi a4, a4, %lo(SHARED_ANCHOR)
  slli a1, a0, 3
  add a1, a1, a4
  ld a1, 0(a1)
  jr a1
.Lfirst0:
  li a0, 30
  ret
.Lfirst1:
  li a0, 31
  ret
  .size shared_first, .-shared_first

  .globl shared_second
  .type shared_second, @function
  .p2align 2
shared_second:
  lui a4, %hi(SHARED_ANCHOR)
  addi a4, a4, %lo(SHARED_ANCHOR)
  slli a1, a0, 3
  add a1, a1, a4
  ld a1, 16(a1)
  jr a1
.Lsecond0:
  li a0, 40
  ret
.Lsecond1:
  li a0, 41
  ret
  .size shared_second, .-shared_second

  .section .rodata,"a",@progbits
  .globl SHARED_ANCHOR
  .type SHARED_ANCHOR, @object
  .p2align 3
SHARED_ANCHOR:
  .dword .Lfirst0
  .dword .Lfirst1
  .size SHARED_ANCHOR, .-SHARED_ANCHOR

  .globl SECOND_JT
  .type SECOND_JT, @object
SECOND_JT:
  .dword .Lsecond0
  .dword .Lsecond1
  .size SECOND_JT, .-SECOND_JT
