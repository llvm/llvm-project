// RUN: %clang %cflags -o %t %s
// RUN: llvm-bolt --strict -o %t.bolt %t
// RUN: llvm-readelf -x .data %t.bolt | FileCheck %s

  .text
  .option norvc
  .globl _start
  .p2align 1
_start:
  // Force BOLT into relocation mode
  .reloc 0, R_RISCV_NONE
  // BOLT removes this nop so the label difference is initially 12 but should be
  // 8 after BOLT processes it.
  nop
  beq x0, x0, _test_end
  addi x1, x1, 1
_test_end:
  ret
  .size _start, .-_start

  .data
// CHECK: Hex dump of section '.data':
// CHECK: 0x{{.*}} 08000000
  .word _test_end - _start
