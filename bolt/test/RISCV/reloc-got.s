// RUN: %clang %cflags -o %t %s
// RUN: llvm-bolt --print-cfg --print-only=_start -o %t.null %t \
// RUN:    | FileCheck %s

  .data
  .globl d
  .p2align 3
d:
  .dword 0

  .globl e
  .p2align 3
e:
  .dword 0

  .text
  .globl _start
  .p2align 1
// CHECK: Binary Function "_start" after building cfg {
_start:
  nop // Here to not make the _start and .Ltmp0 symbols coincide
      // CHECK: auipc t0, %pcrel_hi(__BOLT_got_zero+{{[0-9]+}}) # Label: .Ltmp0
      // CHECK: auipc t1, %pcrel_hi(__BOLT_got_zero+{{[0-9]+}}) # Label: .Ltmp1
      // CHECK-NEXT: ld t0, %pcrel_lo(.Ltmp0)(t0)
      // CHECK-NEXT: ld t1, %pcrel_lo(.Ltmp1)(t1)
1:
  auipc t0, %got_pcrel_hi(d)
2:
  auipc t1, %got_pcrel_hi(e)
  ld t0, %pcrel_lo(1b)(t0)
  ld t1, %pcrel_lo(2b)(t1)
  ret
  .size _start, .-_start