## Verify that RV32C c.jal direct calls are symbolized during disassembly.

# RUN: llvm-mc -triple=riscv32 -mattr=+c -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs %t.o -o %t.exe
# RUN: llvm-bolt %t.exe --print-disasm --print-only=_start -o %t.bolt | FileCheck %s

# CHECK-LABEL: Binary Function "_start" after disassembly {
# CHECK:      00000000: {{.*}}jal callee
# CHECK-NEXT: 00000002:

  .option exact
  .text
  .globl _start, callee
  .type _start, @function
_start:
  c.jal callee
  li a7, 93
  ecall
  .size _start, .-_start

  .type callee, @function
callee:
  ret
  .size callee, .-callee
