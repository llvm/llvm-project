# REQUIRES: mips
# Check R_MIPS_CALL16 relocation calculation.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -shared -o %t.so
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck %s
# RUN: llvm-readelf -A --symbols %t.so | FileCheck -check-prefix=GOT %s

  .text
  .globl  __start
__start:
  lw      $t0,%call16(g1)($gp)

  .globl g1
  .type  g1,@function
g1:
  nop

# CHECK:      Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT: __start:
# CHECK-NEXT:      {{.*}}:  lw  $8, -32744

# GOT: Symbol table '.symtab'
# GOT: {{.*}}:  [[G1:[0-9a-f]+]]  {{.*}} g1

# GOT: Primary GOT:
# GOT:  Global entries:
# GOT:   {{.*}} -32744(gp) [[G1]] [[G1]] FUNC 7 g1
