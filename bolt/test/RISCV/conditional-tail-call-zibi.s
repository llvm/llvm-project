// Check that Zibi conditional branches (beqi/bnei) targeting another
// function are expanded to tail-call blocks, survive block reordering,
// and are emitted with the correct target.

// RUN: llvm-mc -triple riscv64 -mattr=+experimental-zibi -filetype=obj \
// RUN:   -o %t.o %s
// RUN: ld.lld -o %t %t.o
// RUN: llvm-bolt %t -o %t.bolt --reorder-blocks=reverse --print-cfg \
// RUN:   --print-only=conditional_tail_calls 2>&1 | FileCheck %s
// RUN: llvm-objdump -d --disassemble-symbols=conditional_tail_calls \
// RUN:   --mattr=+experimental-zibi %t.bolt | FileCheck %s --check-prefix=DISASM

// CHECK: Binary Function "conditional_tail_calls" after building cfg {
// CHECK:      bnei a0, 0x1, .Ltmp
// CHECK:      beqi a0, 0x1, .Ltmp
// CHECK: BOLT-INFO: basic block reordering modified layout of 1 functions

// DISASM-LABEL: <conditional_tail_calls>:
// DISASM-NEXT: {{.*}} bnei a0, 0x1, {{.*}} <conditional_tail_calls+{{.*}}>
// DISASM-NEXT: {{.*}} j {{.*}} <callee>
// DISASM-NEXT: {{.*}} beqi a0, 0x1, {{.*}} <conditional_tail_calls+{{.*}}>
// DISASM-NEXT: {{.*}} j {{.*}} <callee>
// DISASM-NEXT: {{.*}} ret

  .attribute arch, "rv64i2p1_zibi0p1"
  .text

  .globl conditional_tail_calls
  .type conditional_tail_calls, @function
  .p2align 1
conditional_tail_calls:
  beqi a0, 1, callee
  bnei a0, 1, callee
  ret
  .size conditional_tail_calls, .-conditional_tail_calls

  .globl callee
  .type callee, @function
  .p2align 1
callee:
  ret
  .size callee, .-callee

  .globl _start
  .type _start, @function
  .p2align 1
_start:
  ret
  .size _start, .-_start
