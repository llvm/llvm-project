// RUN: %clang %cflags -o %t %s
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
// CHECK-LABEL: {{^}}.Ltmp0
// CHECK: auipc t0, %pcrel_hi(d) # Label: .Ltmp1
// CHECK-NEXT: ld t0, %pcrel_lo(.Ltmp1)(t0)
// CHECK-NEXT: j .Ltmp0
  nop
1:
  auipc t0, %pcrel_hi(d)
  ld t0, %pcrel_lo(1b)(t0)
  j 1b

/// The local label is used only for %pcrel_lo so no new basic block should
/// start there.
// CHECK-LABEL: {{^}}.LFT0
// CHECK: nop
// CHECK-NEXT: auipc t0, %pcrel_hi(d) # Label: .Ltmp2
// CHECK-NEXT: ld t0, %pcrel_lo(.Ltmp2)(t0)
// CHECK-NEXT: ret
  nop
1:
  auipc t0, %pcrel_hi(d)
  ld t0, %pcrel_lo(1b)(t0)
  ret
  .size _start, .-_start
