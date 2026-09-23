// Check that all base and compressed RISC-V conditional branches targeting
// another function are expanded to tail-call blocks, survive block reordering,
// and are emitted with the correct target.

// RUN: llvm-mc -triple riscv64 -mattr=+c -filetype=obj -o %t.o %s
// RUN: ld.lld -o %t %t.o
// RUN: llvm-bolt %t -o %t.bolt --reorder-blocks=reverse --print-cfg \
// RUN:   --print-only=conditional_tail_calls 2>&1 | FileCheck %s
// RUN: llvm-objdump -d --disassemble-symbols=conditional_tail_calls %t.bolt \
// RUN:   | FileCheck %s --check-prefix=DISASM

// CHECK: Binary Function "conditional_tail_calls" after building cfg {
// CHECK:      beq a0, a1, .LTC0
// CHECK:      bne a0, a1, .LTC1
// CHECK:      blt a0, a1, .LTC2
// CHECK:      bge a0, a1, .LTC3
// CHECK:      bltu a0, a1, .LTC4
// CHECK:      bgeu a0, a1, .LTC5
// CHECK:      beqz a0, .LTC6
// CHECK:      bnez a0, .LTC7
// CHECK: BOLT-INFO: basic block reordering modified layout of 1 functions

// DISASM-LABEL: <conditional_tail_calls>:
// DISASM-NEXT: {{.*}} beq a0, a1, {{.*}} <callee>
// DISASM-NEXT: {{.*}} bne a0, a1, {{.*}} <callee>
// DISASM-NEXT: {{.*}} blt a0, a1, {{.*}} <callee>
// DISASM-NEXT: {{.*}} bge a0, a1, {{.*}} <callee>
// DISASM-NEXT: {{.*}} bltu a0, a1, {{.*}} <callee>
// DISASM-NEXT: {{.*}} bgeu a0, a1, {{.*}} <callee>
// DISASM-NEXT: {{.*}} beqz a0, {{.*}} <callee>
// DISASM-NEXT: {{.*}} bnez a0, {{.*}} <callee>
// DISASM-NEXT: {{.*}} ret

  .text
  .option rvc

  .globl conditional_tail_calls
  .type conditional_tail_calls, @function
  .p2align 1
conditional_tail_calls:
  .option push
  .option exact
  beq a0, a1, callee
  bne a0, a1, callee
  blt a0, a1, callee
  bge a0, a1, callee
  bltu a0, a1, callee
  bgeu a0, a1, callee
  c.beqz a0, callee
  c.bnez a0, callee
  .option pop
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
