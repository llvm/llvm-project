/// Check that functions with internal relocations are considered "simple" and
/// get transformed by BOLT. The tests rely on the "remove-nops" optimization:
/// if nops got removed from the function, it got transformed by BOLT.

// RUN: %clang %cflags -o %t %s
// RUN: llvm-bolt -o %t.bolt %t
// RUN: llvm-objdump -d %t.bolt | FileCheck %s

  .text

  /// These options are only used to make the assembler output easier to predict
  .option norelax
  .option norvc

  .globl _start
  .p2align 1
// CHECK: <_start>:
// CHECK-NEXT: j 0x{{.*}} <_start>
_start:
  nop
1:
  j 1b
  .size _start, .-_start

  .globl f
  .p2align 1
// CHECK: <f>:
// CHECK-NEXT: auipc a0, 0
// CHECK-NEXT: addi a0, a0, 64
f:
  nop
1:
  /// Same as "la a0, g" but more explicit
  auipc a0, %pcrel_hi(g)
  addi  a0, a0, %pcrel_lo(1b)
  ret
  .size f, .-f

  .globl g
  .p2align 1
g:
  ret
  .size g, .-g
