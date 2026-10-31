// RUN: %clang %cflags64 -o %t %s
// RUN: llvm-bolt -o %t.bolt %t
// RUN: llvm-objdump -d --disassemble-symbols=f %t.bolt | FileCheck %s

  .text

  .globl _start
  .p2align 1
_start:
  call f
  .size _start, .-_start
  .globl f
  .p2align 1
// CHECK-LABEL: <f>:
f:
  /// An indirect branch BOLT cannot analyze, which makes the function
  /// non-simple. This stands in for a jump table.
// CHECK-NEXT: beqz a1,
  beqz a1, .Lcont
// CHECK-NEXT: jr a2
  jr a2
.Lcont:
/// The branch to the end of the function must stay a conditional branch.
/// NOTE: This seems not reasonable in general, however this might created when
///       program has __builtin_unreachable or undefined behavior.
// CHECK-NEXT: beqz a0, {{.*}} <g>
  beqz a0, .Lend
// CHECK-NEXT: li a0, 0x1
  li a0, 1
// CHECK-NEXT: ret
  ret
.Lend:
  .size f, .-f

  .globl g
  .p2align 1
g:
  ret
  .size g, .-g
