// Check that FixRISCVCalls rewrites an AUIPC/JALR pair even when a branch
// target splits the pair across two basic blocks.

// RUN: llvm-mc -triple riscv64 -mattr=+c -filetype=obj -o %t.o %s
// RUN: ld.lld --emit-relocs -o %t %t.o
// RUN: llvm-bolt --print-cfg --print-fix-riscv-calls --print-only=_start \
// RUN:   -o %t.bolt %t | FileCheck %s
// RUN: llvm-objdump -d %t.bolt | FileCheck --check-prefix=OBJDUMP %s

  .text
  .option norvc

  .globl target
  .p2align 2
target:
  ret
  .size target, .-target

  .globl _start
  .p2align 2
_start:
  // This branch is never taken, but makes .Ljalr a basic-block entry.
  bne zero, zero, .Ljalr
.Lcall:
  auipc ra, 0
  .reloc .Lcall, R_RISCV_CALL_PLT, target
.Ljalr:
  jalr ra, ra, 0
  ret
  .size _start, .-_start

// CHECK-LABEL: Binary Function "_start" after building cfg {
// CHECK:      auipc ra, target
// CHECK:      jalr

// CHECK-LABEL: Binary Function "_start" after fix-riscv-calls {
// CHECK:      nop
// CHECK:      call target

// OBJDUMP-LABEL: <_start>:
// OBJDUMP:       nop
// OBJDUMP-NEXT:  jal
