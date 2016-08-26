# REQUIRES: mips
# Check LA25 stubs creation. This stub code is necessary when
# non-PIC code calls PIC function.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:   %p/Inputs/mips-pic.s -o %t-pic.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t-npic.o
# RUN: ld.lld %t-npic.o %t-pic.o %p/Inputs/mips-sto-pic.o -o %t.exe
# RUN: llvm-objdump -d %t.exe | FileCheck %s

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
#CHECK:       fpic:
#CHECK-NEXT:    20080:	 00 00 00 00 00 00 00 00         ........
#CHECK-NEXT:    20088:	 00 00 00 00 00 00 00 00         

#CHECK:       fnpic:
#CHECK-NEXT:    20090:	 00 00 00 00 00 00 00 00         ........
#CHECK-NEXT:    20098:	 00 00 00 00 00 00 00 00         ........
#CHECK-NEXT:    200a0:	 3c 19 00 02 08 00 80 20         <...... 
#CHECK-NEXT:    200a8:	 27 39 00 80 00 00 00 00         

# Make sure tha thunks are created properly no matter how
# objects are laid out.
#
# RUN: ld.lld %t-pic.o %t-npic.o %p/Inputs/mips-sto-pic.o -o %t.exe
# RUN: llvm-objdump -d %t.exe | FileCheck -check-prefix=REVERSE %s

# REVERSE:      foo1a:
# REVERSE-NEXT:    20000:       00 00 00 00     nop
#
# REVERSE:      foo1b:
# REVERSE-NEXT:    20004:       00 00 00 00     nop
# REVERSE-NEXT:    20008:       3c 19 00 02     lui     $25, 2
# REVERSE-NEXT:    2000c:       08 00 80 00     j       131072 <foo1a>
# REVERSE-NEXT:    20010:       27 39 00 00     addiu   $25, $25, 0
# REVERSE-NEXT:    20014:       00 00 00 00     nop
# REVERSE-NEXT:    20018:       3c 19 00 02     lui     $25, 2
# REVERSE-NEXT:    2001c:       08 00 80 01     j       131076 <foo1b>
# REVERSE-NEXT:    20020:       27 39 00 04     addiu   $25, $25, 4
# REVERSE-NEXT:    20024:       00 00 00 00     nop
# REVERSE-NEXT:    20028:       00 00 00 00     nop
# REVERSE-NEXT:    2002c:       00 00 00 00     nop
#
# REVERSE:      foo2:
# REVERSE-NEXT:    20030:       00 00 00 00     nop
# REVERSE-NEXT:    20034:       3c 19 00 02     lui     $25, 2
# REVERSE-NEXT:    20038:       08 00 80 0c     j       131120 <foo2>
# REVERSE-NEXT:    2003c:       27 39 00 30     addiu   $25, $25, 48
# REVERSE-NEXT:    20040:       00 00 00 00     nop
# REVERSE-NEXT:    20044:       00 00 00 00     nop
# REVERSE-NEXT:    20048:       00 00 00 00     nop
# REVERSE-NEXT:    2004c:       00 00 00 00     nop
#
# REVERSE:      __start:
# REVERSE-NEXT:    20050:       0c 00 80 02     jal     131080 <foo1b+0x4>
# REVERSE-NEXT:    20054:       00 00 00 00     nop
# REVERSE-NEXT:    20058:       0c 00 80 0d     jal     131124 <foo2+0x4>
# REVERSE-NEXT:    2005c:       00 00 00 00     nop
# REVERSE-NEXT:    20060:       0c 00 80 06     jal     131096 <foo1b+0x14>
# REVERSE-NEXT:    20064:       00 00 00 00     nop
# REVERSE-NEXT:    20068:       0c 00 80 0d     jal     131124 <foo2+0x4>
# REVERSE-NEXT:    2006c:       00 00 00 00     nop
# REVERSE-NEXT:    20070:       0c 00 80 28     jal     131232 <fnpic+0x10>
# REVERSE-NEXT:    20074:       00 00 00 00     nop
# REVERSE-NEXT:    20078:       0c 00 80 24     jal     131216 <fnpic>
# REVERSE-NEXT:    2007c:       00 00 00 00     nop
#
#REVERSE:       fpic:
#REVERSE-NEXT:    20080:	 00 00 00 00 00 00 00 00         ........
#REVERSE-NEXT:    20088:	 00 00 00 00 00 00 00 00         
#
#REVERSE:       fnpic:
#REVERSE-NEXT:    20090:	 00 00 00 00 00 00 00 00         ........
#REVERSE-NEXT:    20098:	 00 00 00 00 00 00 00 00         ........
#REVERSE-NEXT:    200a0:	 3c 19 00 02 08 00 80 20         <...... 
#REVERSE-NEXT:    200a8:	 27 39 00 80 00 00 00 00         

  .text
  .globl __start
__start:
  jal foo1a
  jal foo2
  jal foo1b
  jal foo2
  jal fpic
  jal fnpic
