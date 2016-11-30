# Check that the linker use a value of _gp symbol defined
# in a linker script to calculate GOT relocations.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: echo "SECTIONS { \
# RUN:          .text : { *(.text) } \
# RUN:          _gp = . + 0x100; \
# RUN:          .got  : { *(.got) } }" > %t.script
# RUN: ld.lld -shared -o %t.so --script %t.script %t.o
# RUN: llvm-objdump -s -t %t.so | FileCheck %s

# REQUIRES: mips

# CHECK:      Contents of section .text:
# CHECK-NEXT:  0000 3c080000 2108010c 8f82ffe4
#                   ^-- %hi(_gp_disp)
#                            ^-- %lo(_gp_disp)
#                                     ^-- 8 - (0x10c - 0xe8)
#                                         G - (GP - .got)

# CHECK:      Contents of section .reginfo:
# CHECK-NEXT:  0028 10000104 00000000 00000000 00000000
# CHECK-NEXT:  0038 00000000 0000010c
#                            ^-- _gp

# CHECK:      Contents of section .data:
# CHECK-NEXT:  0100 fffffef4
#                   ^-- 0-0x10c

# CHECK: 00000000         .text           00000000 foo
# CHECK: 0000010c         .got            00000000 .hidden _gp_disp
# CHECK: 0000010c         .text           00000000 .hidden _gp

  .text
foo:
  lui    $t0, %hi(_gp_disp)
  addi   $t0, $t0, %lo(_gp_disp)
  lw     $v0, %call16(bar)($gp)

  .data
  .gpword foo
