// RUN: %clang %cflags64 -o %t %s
// RUN: llvm-bolt --print-cfg --print-only=_start -o %t.null %t \
// RUN:    | FileCheck %s

  .data
  .globl d
  .p2align 3
d:
  .dword 0

  .text
  .globl _start
  .p2align 1
// CHECK-LABEL: Binary Function "_start" after building cfg {
_start:
/// The local label is used for %pcrel_lo as well as a jump target so a new
/// basic block should start there.
// CHECK-LABEL: {{^}}.LBB00
// CHECK: nop
// CHECK: {{^}}[[BRANCH_LABEL:.Ltmp[0-9]+]]
// CHECK: auipc t0, %pcrel_hi(d) # Label: [[HI_LABEL:.Ltmp[0-9]+]]
// CHECK-NEXT: ld t0, %pcrel_lo([[HI_LABEL]])(t0)
// CHECK-NEXT: j [[BRANCH_LABEL]]
  nop
1:
  auipc t0, %pcrel_hi(d)
  ld t0, %pcrel_lo(1b)(t0)
  j 1b

/// The local label is used only for %pcrel_lo so no new basic block should
/// start there.
// CHECK-LABEL: {{^}}.LFT0
// CHECK: nop
// CHECK-NEXT: auipc t0, %pcrel_hi(d) # Label: [[SECOND_HI:.Ltmp[0-9]+]]
// CHECK-NEXT: ld t0, %pcrel_lo([[SECOND_HI]])(t0)
// CHECK-NEXT: ret
  nop
1:
  auipc t0, %pcrel_hi(d)
  ld t0, %pcrel_lo(1b)(t0)
  ret
  .size _start, .-_start
