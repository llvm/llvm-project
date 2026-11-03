## Check that BOLT parses its internal address map correctly for a 32-bit
## target. Address-map entries contain 64-bit values even when the target code
## pointer size is 32 bits.

# RUN: llvm-mc -triple=riscv32 -mattr=+c -filetype=obj %s -o %t.o
# RUN: ld.lld -q %t.o -o %t
# RUN: llvm-bolt --enable-bat %t -o %t.bolt
# RUN: llvm-bat-dump %t.bolt --dump-all 2>&1 | \
# RUN:   FileCheck %s --check-prefix=CHECK-BAT

# CHECK-BAT:      BOLT-INFO: Parsed 1 BAT entries
# CHECK-BAT:      BAT tables for 1 functions:
# CHECK-BAT:      Function Address: 0x{{[0-9a-f]+}}, hash:
# CHECK-BAT-NEXT: BB mappings:
# CHECK-BAT-NEXT: 0x0 -> 0x0 hash:
# CHECK-BAT-NEXT: 0x0 -> 0x2 hash:
# CHECK-BAT-NEXT: NumBlocks: 2

  .text
  .globl _start
  .type _start, @function
  .p2align 1
_start:
  nop
1:
  j 1b
  .size _start, .-_start
