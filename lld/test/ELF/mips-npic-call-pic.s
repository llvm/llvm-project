# Check LA25 stubs creation. This stub code is necessary when
# non-PIC code calls PIC function.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:   %p/Inputs/mips-pic.s -o %t-pic.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t-npic.o
# RUN: ld.lld %t-npic.o %t-pic.o -o %t.exe
# RUN: llvm-objdump -d %t.exe | FileCheck %s

# REQUIRES: mips

# CHECK:     Disassembly of section .text:
# CHECK-NEXT: __start:
# CHECK-NEXT:    20000:       0c 00 80 0a     jal     131112
#                                                     ^-- 0x20030 .pic.foo1a
# CHECK-NEXT:    20004:       00 00 00 00     nop
# CHECK-NEXT:    20008:       0c 00 80 15     jal     131156
#                                                     ^-- 0x20060 .pic.foo2
# CHECK-NEXT:    2000c:       00 00 00 00     nop
# CHECK-NEXT:    20010:       0c 00 80 0e     jal     131128
#                                                     ^-- 0x20040 .pic.foo1b
# CHECK-NEXT:    20014:       00 00 00 00     nop
# CHECK-NEXT:    20018:       0c 00 80 15     jal     131156
#                                                     ^-- 0x20060 .pic.foo2
# CHECK-NEXT:    2001c:       00 00 00 00     nop
#
# CHECK:      foo1a:
# CHECK-NEXT:    20020:       00 00 00 00     nop
#
# CHECK:      foo1b:
# CHECK-NEXT:    20024:       00 00 00 00     nop
#
# CHECK-NEXT:    20028:       3c 19 00 02     lui     $25, 2
# CHECK-NEXT:    2002c:       08 00 80 08     j       131104 <foo1a>
# CHECK-NEXT:    20030:       27 39 00 20     addiu   $25, $25, 32
# CHECK-NEXT:    20034:       00 00 00 00     nop
# CHECK-NEXT:    20038:       3c 19 00 02     lui     $25, 2
# CHECK-NEXT:    2003c:       08 00 80 09     j       131108 <foo1b>
# CHECK-NEXT:    20040:       27 39 00 24     addiu   $25, $25, 36
# CHECK-NEXT:    20044:       00 00 00 00     nop
# CHECK-NEXT:    20048:       00 00 00 00     nop
# CHECK-NEXT:    2004c:       00 00 00 00     nop
#
# CHECK:      foo2:
# CHECK-NEXT:    20050:       00 00 00 00     nop
#
# CHECK-NEXT:    20054:       3c 19 00 02     lui     $25, 2
# CHECK-NEXT:    20058:       08 00 80 14     j       131152 <foo2>
# CHECK-NEXT:    2005c:       27 39 00 50     addiu   $25, $25, 80
# CHECK-NEXT:    20060:       00 00 00 00     nop

  .text
  .globl __start
__start:
  jal foo1a
  jal foo2
  jal foo1b
  jal foo2
