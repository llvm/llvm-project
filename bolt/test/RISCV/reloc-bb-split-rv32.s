/// RV32 variant of reloc-bb-split.s — uses lw instead of ld.

// RUN: llvm-mc -triple riscv32 -mattr=+c -filetype=obj -o %t.o %s
// RUN: ld.lld -q -o %t %t.o
// RUN: llvm-bolt --print-cfg --print-only=_start -o %t.null %t \
// RUN:    | FileCheck %s

  .data
  .globl d
  .p2align 2
d:
  .word 0

  .text
  .globl _start
  .p2align 1
// CHECK-LABEL: Binary Function "_start" after building cfg {
_start:
/// The local label is used for %pcrel_lo as well as a jump target so a new
/// basic block should start there.
// CHECK-LABEL: {{^}}.LBB00
// CHECK: nop
// CHECK-LABEL: {{^}}.Ltmp0
// CHECK: auipc t0, %pcrel_hi(d)
// CHECK-NEXT: lw t0, %pcrel_lo({{.*}})(t0)
// CHECK-NEXT: j .Ltmp0
  nop
1:
  auipc t0, %pcrel_hi(d)
  lw t0, %pcrel_lo(1b)(t0)
  j 1b

/// The local label is used only for %pcrel_lo so no new basic block should
/// start there.
// CHECK-LABEL: {{^}}.LFT0
// CHECK: nop
// CHECK: auipc t0, %pcrel_hi(d)
// CHECK-NEXT: lw t0, %pcrel_lo({{.*}})(t0)
// CHECK-NEXT: ret
  nop
1:
  auipc t0, %pcrel_hi(d)
  lw t0, %pcrel_lo(1b)(t0)
  ret
  .size _start, .-_start
