// RUN: %clang %cflags64 -o %t %s
// RUN: llvm-bolt --check-encoding --print-cfg --print-only=_start \
// RUN:    -o %t.null %t \
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

/// An unrelated instruction can sit between the AUIPC and its load. The
/// symbolizer locates the low relocation through its reference to the AUIPC.
// CHECK:      auipc t1, %pcrel_hi(__BOLT_got_zero+[[GOT]]) # Label: [[HI2:\.Ltmp[0-9]+]]
// CHECK-NEXT: addi t2, t2, 0x7ff
// CHECK-NEXT: ld t1, %pcrel_lo([[HI2]])(t1)
2:
  auipc t1, %got_pcrel_hi(d)
  addi t2, t2, 2047
  ld t1, %pcrel_lo(2b)(t1)
  j .L1
.L2:
// CHECK:      ld t1, %pcrel_lo([[HI3:\.Ltmp[0-9]+]])(t1)
// CHECK-NEXT: j
  ld t1, %pcrel_lo(3f)(t1)
  j .Lexit
.L1:
  nop
/// The low relocation can also precede the AUIPC in output basic-block order.
// CHECK:      nop
// CHECK-NEXT: auipc t1, %pcrel_hi(__BOLT_got_zero+[[GOT]]) # Label: [[HI3]]
// CHECK-NEXT: j
3:
  auipc t1, %got_pcrel_hi(d)
  j .L2
.Lexit:
  ret
  .size _start, .-_start
