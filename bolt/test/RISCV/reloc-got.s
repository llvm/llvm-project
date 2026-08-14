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
// CHECK: Binary Function "_start" after building cfg {
_start:
  nop // Here to not make the _start and the %pcrel_lo label coincide

/// The load follows the AUIPC, so BOLT resolves the right GOT entry.
// CHECK:      auipc t0, %pcrel_hi(__BOLT_got_zero+[[GOT:[0-9]+]]) # Label: [[HI:\.Ltmp[0-9]+]]
// CHECK-NEXT: ld t0, %pcrel_lo([[HI]])(t0)
1:
  auipc t0, %got_pcrel_hi(d)
  ld t0, %pcrel_lo(1b)(t0)

/// An unrelated instruction sits between the AUIPC and its load.
// FIXME: The AUIPC below should also use __BOLT_got_zero+[[GOT]], but BOLT
// takes the low part from the ADDI instead of from the load that names the
// AUIPC's label.
// CHECK-NOT:  __BOLT_got_zero+[[GOT]])
// CHECK:      addi t2, t2, 0x7ff
// CHECK-NEXT: ld t1, %pcrel_lo({{\.Ltmp[0-9]+}})(t1)
2:
  auipc t1, %got_pcrel_hi(d)
  addi t2, t2, 2047
  ld t1, %pcrel_lo(2b)(t1)
  j .L1
.L2:
  ld t1, %pcrel_lo(3f)(t1)
  j .Lexit
.L1:
  nop
/// The load lives in another basic block, so nothing follows the AUIPC but
/// the terminator.
// FIXME: The AUIPC below should also use __BOLT_got_zero+[[GOT]], but BOLT
// takes the low part from the jump.
// CHECK:      nop
// CHECK-NOT:  __BOLT_got_zero+[[GOT]])
// CHECK:      j
3:
  auipc t1, %got_pcrel_hi(d)
  j .L2
.Lexit:
  ret
  .size _start, .-_start
