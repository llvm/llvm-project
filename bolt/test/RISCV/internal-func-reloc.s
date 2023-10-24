/// Check that functions with internal relocations are considered "simple" and
/// get transformed by BOLT. The tests rely on the "remove-nops" optimization:
/// if nops got removed from the function, it got transformed by BOLT.

// RUN: llvm-mc -triple riscv64 -filetype=obj -o %t.o %s
// RUN: ld.lld --emit-relocs -o %t %t.o
// RUN: llvm-bolt -o %t.bolt %t
// RUN: llvm-objdump -d %t.bolt | FileCheck %s

  .text
  .globl _start
  .p2align 2
// CHECK: <_start>:
// CHECK-NEXT: j 0x{{.*}} <_start>
_start:
  nop
1:
  j 1b
  .size _start, .-_start

  .globl f
  .p2align 2
// CHECK: <f>:
// CHECK-NEXT: auipc a0, [[#]]
// CHECK-NEXT: addi a0, a0, [[#]]
f:
  nop
1:
  /// Same as "la a0, g" but more explicit
  auipc a0, %pcrel_hi(g)
  addi  a0, a0, %pcrel_lo(1b)
  ret
  .size f, .-f

  .globl g
  .p2align 2
g:
  ret
  .size g, .-g
