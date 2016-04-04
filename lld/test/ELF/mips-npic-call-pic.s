# Check LA25 stubs creation. This stub code is necessary when
# non-PIC code calls PIC function.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:   %p/Inputs/mips-pic.s -o %t-pic.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t-npic.o
# RUN: ld.lld %t-npic.o %t-pic.o %p/Inputs/mips-sto-pic.o -o %t.exe
# RUN: llvm-objdump -d %t.exe | FileCheck %s

# REQUIRES: mips

# CHECK:     Disassembly of section .text:
# CHECK-NEXT: __start:
# CHECK-NEXT:    20000:       0c 00 80 0e     jal     131128 <foo1b+0x4>
#                                                            ^-- .pic.foo1a
# CHECK-NEXT:    20004:       00 00 00 00     nop
# CHECK-NEXT:    20008:       0c 00 80 19     jal     131172 <foo2+0x4>
#                                                            ^-- .pic.foo2
# CHECK-NEXT:    2000c:       00 00 00 00     nop
# CHECK-NEXT:    20010:       0c 00 80 12     jal     131144 <foo1b+0x14>
#                                                            ^-- .pic.foo1b
# CHECK-NEXT:    20014:       00 00 00 00     nop
# CHECK-NEXT:    20018:       0c 00 80 19     jal     131172 <foo2+0x4>
#                                                            ^-- .pic.foo2
# CHECK-NEXT:    2001c:       00 00 00 00     nop
# CHECK-NEXT:    20020:       0c 00 80 28     jal     131232 <fnpic+0x10>
#                                                            ^-- .pic.fpic
# CHECK-NEXT:    20024:       00 00 00 00     nop
# CHECK-NEXT:    20028:       0c 00 80 24     jal     131216 <fnpic>
# CHECK-NEXT:    2002c:       00 00 00 00     nop
#
# CHECK:      foo1a:
# CHECK-NEXT:    20030:       00 00 00 00     nop
#
# CHECK:      foo1b:
# CHECK-NEXT:    20034:       00 00 00 00     nop
#
# CHECK-NEXT:    20038:       3c 19 00 02     lui     $25, 2
# CHECK-NEXT:    2003c:       08 00 80 0c     j       131120 <foo1a>
# CHECK-NEXT:    20040:       27 39 00 30     addiu   $25, $25, 48
# CHECK-NEXT:    20044:       00 00 00 00     nop
# CHECK-NEXT:    20048:       3c 19 00 02     lui     $25, 2
# CHECK-NEXT:    2004c:       08 00 80 0d     j       131124 <foo1b>
# CHECK-NEXT:    20050:       27 39 00 34     addiu   $25, $25, 52
# CHECK-NEXT:    20054:       00 00 00 00     nop
# CHECK-NEXT:    20058:       00 00 00 00     nop
# CHECK-NEXT:    2005c:       00 00 00 00     nop
#
# CHECK:      foo2:
# CHECK-NEXT:    20060:       00 00 00 00     nop
#
# CHECK-NEXT:    20064:       3c 19 00 02     lui     $25, 2
# CHECK-NEXT:    20068:       08 00 80 18     j       131168 <foo2>
# CHECK-NEXT:    2006c:       27 39 00 60     addiu   $25, $25, 96
# CHECK-NEXT:    20070:       00 00 00 00     nop
# CHECK-NEXT:    20074:       00 00 00 00     nop
# CHECK-NEXT:    20078:       00 00 00 00     nop
# CHECK-NEXT:    2007c:       00 00 00 00     nop
#
# CHECK:      fpic:
# CHECK-NEXT:    20080:       00 00 00 00     nop
# CHECK-NEXT:    20084:       00 00 00 00     nop
# CHECK-NEXT:    20088:       00 00 00 00     nop
# CHECK-NEXT:    2008c:       00 00 00 00     nop
#
# CHECK:      fnpic:
# CHECK-NEXT:    20090:       00 00 00 00     nop
# CHECK-NEXT:    20094:       00 00 00 00     nop
# CHECK-NEXT:    20098:       00 00 00 00     nop
# CHECK-NEXT:    2009c:       00 00 00 00     nop
# CHECK-NEXT:    200a0:       3c 19 00 02     lui     $25, 2
# CHECK-NEXT:    200a4:       08 00 80 20     j       131200 <fpic>
# CHECK-NEXT:    200a8:       27 39 00 80     addiu   $25, $25, 128

  .text
  .globl __start
__start:
  jal foo1a
  jal foo2
  jal foo1b
  jal foo2
  jal fpic
  jal fnpic
