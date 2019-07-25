# REQUIRES: mips
# Check R_MIPS_CALL16 relocation calculation.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -shared -o %t.so
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck %s
# RUN: llvm-readobj --mips-plt-got --symbols %t.so \
# RUN:   | FileCheck -check-prefix=GOT %s

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
# CHECK-NEXT:      10000:       lw   $8, -32744

# GOT:      Name: g1
# GOT-NEXT: Value: 0x[[ADDR:[0-9A-F]+]]

# GOT:      Local entries [
# GOT-NEXT: ]
# GOT-NEXT: Global entries [
# GOT-NEXT:   Entry {
# GOT-NEXT:     Address:
# GOT-NEXT:     Access: -32744
# GOT-NEXT:     Initial: 0x[[ADDR]]
# GOT-NEXT:     Value: 0x[[ADDR]]
# GOT-NEXT:     Type: Function
# GOT-NEXT:     Section: .text
# GOT-NEXT:     Name: g1
# GOT-NEXT:   }
# GOT-NEXT: ]
