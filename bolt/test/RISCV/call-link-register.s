// RUN: %clang %cflags64 -Wl,--no-relax -o %t %s
// RUN: llvm-bolt -o %t.bolt %t
// RUN: llvm-objdump -d --disassemble-symbols=_start %t.bolt | FileCheck %s

  .text

  .globl _start
  .p2align 1
_start:
// CHECK-LABEL: <_start>:
/// The auipc of the pair, replaced by a nop once the call is rewritten.
// CHECK-NEXT: nop
// CHECK-NEXT: jal t0, 0x{{.*}} <f>
  call t0, f
/// A jal in direct range is rewritten on its own, with no auipc to nop out.
// CHECK-NEXT: jal t0, 0x{{.*}} <f>
  jal t0, f
/// A call that already links through ra keeps ra.
// CHECK-NEXT: nop
// CHECK-NEXT: jal 0x{{.*}} <f>
  call f
// CHECK-NEXT: jal 0x{{.*}} <f>
  jal ra, f
  ret
  .size _start, .-_start

  .globl f
  .p2align 1
f:
  jr t0
  .size f, .-f
