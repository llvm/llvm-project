// A conditional branch from f to g is a tail call: g must return to _start
// without overwriting ra. CFG normalization creates a new tail instruction,
// which must be marked as a tail call before FixRISCVCallsPass rewrites it.

// RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o
// RUN: ld.lld --emit-relocs --no-relax %t.o -o %t
// RUN: llvm-bolt %t -o %t.bolt --enable-bat --print-cfg \
// RUN:   --print-fix-riscv-calls --print-only=f \
// RUN:   --simplify-conditional-tail-calls=false | FileCheck %s
// RUN: llvm-objdump -d -M no-aliases --disassemble-symbols=f %t.bolt \
// RUN:   | FileCheck %s --check-prefix=DISASM

// --enable-bat keeps Offset annotations so their preservation is also checked.
// CHECK-LABEL: Binary Function "f" after building cfg {
// CHECK: tail g # TAILCALL # Offset: 0
// CHECK-LABEL: Binary Function "f" after fix-riscv-calls {
// CHECK: tail g # TAILCALL # Offset: 0

// DISASM-LABEL: <f>:
// DISASM-NOT: jal ra,
// DISASM: jal zero, {{.*}}<g>
// DISASM-NOT: jal ra,

  .text
  .option norvc
  .globl _start, f, g

  .type _start, @function
_start:
  li a0, 0
  call f
  li a7, 93
  ecall
  .size _start, .-_start

  .type f, @function
f:
  .option push
  .option exact
  beq a0, zero, g
  .option pop
  ret
  .size f, .-f

  .type g, @function
g:
  ret
  .size g, .-g
