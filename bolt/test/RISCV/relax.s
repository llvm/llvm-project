// RUN: llvm-mc -triple riscv64 -mattr=+c,+relax -filetype obj -o %t.o %s
// RUN: ld.lld --emit-relocs -o %t %t.o
// RUN: llvm-bolt --print-cfg --print-fix-riscv-calls --print-only=_start \
// RUN:     -o %t.bolt %t | FileCheck %s
// RUN: llvm-objdump -d %t.bolt | FileCheck --check-prefix=OBJDUMP %s

// CHECK:      Binary Function "_start" after building cfg {
// CHECK:      jal ra, near_f
// CHECK-NEXT: auipc ra, far_f@plt
// CHECK-NEXT: jalr ra, 12(ra)
// CHECK-NEXT: j near_f

// CHECK:      Binary Function "_start" after fix-riscv-calls {
// CHECK:      call near_f
// CHECK-NEXT: call far_f
// CHECK-NEXT: tail near_f

// OBJDUMP:      0000000000600000 <_start>:
// OBJDUMP-NEXT:     jal 0x600040 <near_f>
// OBJDUMP-NEXT:     auipc ra, 512
// OBJDUMP-NEXT:     jalr 124(ra)
// OBJDUMP-NEXT:     j 0x600040 <near_f>
// OBJDUMP:      0000000000600040 <near_f>:
// OBJDUMP:      0000000000800080 <far_f>:

  .text
  .globl _start
  .p2align 1
_start:
  call near_f
  call far_f
  tail near_f
  .size _start, .-_start

  .global near_f
  .p2align 1
near_f:
  ret
  .size near_f, .-near_f

  .skip (1 << 21)

  .global far_f
  .p2align 1
far_f:
  ret
  .size far_f, .-far_f
