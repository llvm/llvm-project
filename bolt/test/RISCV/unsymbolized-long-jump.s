// Test that a relocation-free AUIPC/JALR no-link pair targeting the same
// function is not misclassified as a tail call.

// RUN: llvm-mc -triple riscv64 -mattr=-relax -filetype=obj -o %t.o %s
// RUN: ld.lld --no-relax --emit-relocs -e _start -o %t %t.o
// RUN: llvm-bolt --print-cfg --print-only=_start -o %t.bolt %t \
// RUN:     | FileCheck --check-prefix=BOLT %s
// RUN: llvm-objdump -d %t.bolt | FileCheck --check-prefix=OBJDUMP %s

// BOLT-LABEL: Binary Function "_start" after building cfg {
// BOLT:       auipc t1, 0x0
// BOLT-NEXT:  jr 0xc(t1)

// OBJDUMP-LABEL: <_start>:
// OBJDUMP:       auipc t1, 0x0
// OBJDUMP-NEXT:  jr 0xc(t1)

  .text
  .option norvc
  .option norelax

  .globl _start
  .type _start,@function
_start:
  auipc t1, 0
  jalr zero, 12(t1)
  addi a0, a0, 1
.Ltarget:
  ret
  .size _start, .-_start

  // Retain a relocation so BOLT processes the executable in relocation mode.
  .globl relocated_call
  .type relocated_call,@function
relocated_call:
  call _start
  ret
  .size relocated_call, .-relocated_call
