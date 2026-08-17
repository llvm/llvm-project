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
/// FIXME: the link register is dropped here: the call is rewritten to link
/// through ra, so f returns to whatever t0 happens to hold.
// CHECK-NEXT: jal 0x{{.*}} <f>
  call t0, f
  ret
  .size _start, .-_start

  .globl f
  .p2align 1
f:
  jr t0
  .size f, .-f
