# REQUIRES: mips
# Check LA25 stubs creation. This stub code is necessary when
# non-PIC code calls PIC function.
# RUN: echo "SECTIONS { .out 0x20000 : { *(.text.*) . = . + 0x100 ;  *(.text) }  }" > %t1.script
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:   %p/Inputs/mips-fpic.s -o %t-fpic.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:   %p/Inputs/mips-fnpic.s -o %t-fnpic.o
# RUN: ld.lld -r %t-fpic.o %t-fnpic.o -o %t-sto-pic.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:   %p/Inputs/mips-pic.s -o %t-pic.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t-npic.o
# RUN: ld.lld --script %t1.script %t-npic.o %t-pic.o %t-sto-pic.o -o %t.exe
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.exe | FileCheck %s

# CHECK: Disassembly of section .out:
# CHECK-EMPTY:
# CHECK-NEXT: <__LA25Thunk_foo1a>:
# CHECK-NEXT:    20000:       lui     $25, 2
# CHECK-NEXT:    20004:       j       {{.*}} <foo1a>
# CHECK-NEXT:    20008:       addiu   $25, $25, 32
# CHECK-NEXT:    2000c:       nop

# CHECK: <__LA25Thunk_foo1b>:
# CHECK-NEXT:    20010:       lui     $25, 2
# CHECK-NEXT:    20014:       j       {{.*}} <foo1b>
# CHECK-NEXT:    20018:       addiu   $25, $25, 36
# CHECK-NEXT:    2001c:       nop

# CHECK: <foo1a>:
# CHECK-NEXT:    20020:       nop

# CHECK: <foo1b>:
# CHECK-NEXT:    20024:       nop

# CHECK: <__LA25Thunk_foo2>:
# CHECK-NEXT:    20028:       lui     $25, 2
# CHECK-NEXT:    2002c:       j       {{.*}} <foo2>
# CHECK-NEXT:    20030:       addiu   $25, $25, 64
# CHECK-NEXT:    20034:       nop

# CHECK: <foo2>:
# CHECK-NEXT:    20040:       nop

# CHECK: <__start>:
# CHECK-NEXT:    20150:       jal     {{.*}} <__LA25Thunk_foo1a>
# CHECK-NEXT:    20154:       nop
# CHECK-NEXT:    20158:       jal     {{.*}} <__LA25Thunk_foo2>
# CHECK-NEXT:    2015c:       nop
# CHECK-NEXT:    20160:       jal     {{.*}} <__LA25Thunk_foo1b>
# CHECK-NEXT:    20164:       nop
# CHECK-NEXT:    20168:       jal     {{.*}} <__LA25Thunk_foo2>
# CHECK-NEXT:    2016c:       nop
# CHECK-NEXT:    20170:       jal     {{.*}} <__LA25Thunk_fpic>
# CHECK-NEXT:    20174:       nop
# CHECK-NEXT:    20178:       jal     {{.*}} <fnpic>
# CHECK-NEXT:    2017c:       nop

# CHECK: <__LA25Thunk_fpic>:
# CHECK-NEXT:    20180:       lui     $25, 2
# CHECK-NEXT:    20184:       j       {{.*}} <fpic>
# CHECK-NEXT:    20188:       addiu   $25, $25, 400
# CHECK-NEXT:    2018c:       nop

# CHECK: <fpic>:
# CHECK-NEXT:    20190:       nop

# CHECK: <fnpic>:
# CHECK-NEXT:    201a0:       nop

  .text
  .globl __start
__start:
  jal foo1a
  jal foo2
  jal foo1b
  jal foo2
  jal fpic
  jal fnpic

# Test script with orphans added to existing OutputSection, the .text.1 and
# .text.2 sections will be added to .text
# RUN: echo "SECTIONS { .text 0x20000 : { *(.text) }  }" > %t2.script
# RUN: ld.lld --script %t2.script %t-npic.o %t-pic.o %t-sto-pic.o -o %t2.exe
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t2.exe | FileCheck --check-prefix=ORPH1 %s

# ORPH1: Disassembly of section .text:
# ORPH1-EMPTY:
# ORPH1-NEXT: <__start>:
# ORPH1-NEXT:    20000:       jal     {{.*}} <__LA25Thunk_foo1a>
# ORPH1-NEXT:    20004:       nop
# ORPH1-NEXT:    20008:       jal     {{.*}} <__LA25Thunk_foo2>
# ORPH1-NEXT:    2000c:       nop
# ORPH1-NEXT:    20010:       jal     {{.*}} <__LA25Thunk_foo1b>
# ORPH1-NEXT:    20014:       nop
# ORPH1-NEXT:    20018:       jal     {{.*}} <__LA25Thunk_foo2>
# ORPH1-NEXT:    2001c:       nop
# ORPH1-NEXT:    20020:       jal     {{.*}} <__LA25Thunk_fpic>
# ORPH1-NEXT:    20024:       nop
# ORPH1-NEXT:    20028:       jal     {{.*}} <fnpic>
# ORPH1-NEXT:    2002c:       nop

# ORPH1: <__LA25Thunk_fpic>:
# ORPH1-NEXT:    20030:       lui     $25, 2
# ORPH1-NEXT:    20034:       j       {{.*}} <fpic>
# ORPH1-NEXT:    20038:       addiu   $25, $25, 64
# ORPH1-NEXT:    2003c:       nop

# ORPH1: <fpic>:
# ORPH1-NEXT:    20040:       nop

# ORPH1: <fnpic>:
# ORPH1-NEXT:    20050:       nop

# ORPH1: <__LA25Thunk_foo1a>:
# ORPH1-NEXT:    20060:       lui     $25, 2
# ORPH1-NEXT:                 j       {{.*}} <foo1a>
# ORPH1-NEXT:                 addiu   $25, $25, 128
# ORPH1-NEXT:                 nop

# ORPH1: <__LA25Thunk_foo1b>:
# ORPH1-NEXT:    20070:       lui     $25, 2
# ORPH1-NEXT:                 j       {{.*}} <foo1b>
# ORPH1-NEXT:                 addiu   $25, $25, 132
# ORPH1-NEXT:                 nop

# ORPH1: <foo1a>:
# ORPH1-NEXT:    20080:       nop

# ORPH1: <foo1b>:
# ORPH1-NEXT:    20084:       nop

# ORPH1: <__LA25Thunk_foo2>:
# ORPH1-NEXT:    20090:       lui     $25, 2
# ORPH1-NEXT:                 j       {{.*}} <foo2>
# ORPH1-NEXT:                 addiu   $25, $25, 160
# ORPH1-NEXT:                 nop

# ORPH1: <foo2>:
# ORPH1-NEXT:    200a0:       nop

# Test script with orphans added to new OutputSection, the .text.1 and
# .text.2 sections will form a new OutputSection .text
# RUN: echo "SECTIONS { .out 0x20000 : { *(.text) } .text : {*(.text*)} }" > %t3.script
# RUN: ld.lld --script %t3.script %t-npic.o %t-pic.o %t-sto-pic.o -o %t3.exe
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t3.exe | FileCheck --check-prefix=ORPH2 %s

# ORPH2: Disassembly of section .out:
# ORPH2-EMPTY:
# ORPH2-NEXT: <__start>:
# ORPH2-NEXT:    20000:       jal     {{.*}} <__LA25Thunk_foo1a>
# ORPH2-NEXT:    20004:       nop
# ORPH2-NEXT:    20008:       jal     {{.*}} <__LA25Thunk_foo2>
# ORPH2-NEXT:    2000c:       nop
# ORPH2-NEXT:    20010:       jal     {{.*}} <__LA25Thunk_foo1b>
# ORPH2-NEXT:    20014:       nop
# ORPH2-NEXT:    20018:       jal     {{.*}} <__LA25Thunk_foo2>
# ORPH2-NEXT:    2001c:       nop
# ORPH2-NEXT:    20020:       jal     {{.*}} <__LA25Thunk_fpic>
# ORPH2-NEXT:    20024:       nop
# ORPH2-NEXT:    20028:       jal     {{.*}} <fnpic>
# ORPH2-NEXT:    2002c:       nop

# ORPH2: <__LA25Thunk_fpic>:
# ORPH2-NEXT:    20030:       lui     $25, 2
# ORPH2-NEXT:    20034:       j       {{.*}} <fpic>
# ORPH2-NEXT:    20038:       addiu   $25, $25, 64
# ORPH2-NEXT:    2003c:       nop

# ORPH2: <fpic>:
# ORPH2-NEXT:    20040:       nop

# ORPH2: <fnpic>:
# ORPH2-NEXT:    20050:       nop
# ORPH2-EMPTY:
# ORPH2-NEXT: Disassembly of section .text:
# ORPH2-EMPTY:

# ORPH2-NEXT: <__LA25Thunk_foo1a>:
# ORPH2-NEXT:    20060:       lui     $25, 2
# ORPH2-NEXT:    20064:       j       {{.*}} <foo1a>
# ORPH2-NEXT:    20068:       addiu   $25, $25, 128
# ORPH2-NEXT:    2006c:       nop

# ORPH2: <__LA25Thunk_foo1b>:
# ORPH2-NEXT:    20070:       lui     $25, 2
# ORPH2-NEXT:    20074:       j       {{.*}} <foo1b>
# ORPH2-NEXT:    20078:       addiu   $25, $25, 132
# ORPH2-NEXT:    2007c:       nop

# ORPH2: <foo1a>:
# ORPH2-NEXT:    20080:       nop

# ORPH2: <foo1b>:
# ORPH2-NEXT:    20084:       nop

# ORPH2: <__LA25Thunk_foo2>:
# ORPH2-NEXT:    20088:       lui     $25, 2
# ORPH2-NEXT:    2008c:       j       {{.*}} <foo2>
# ORPH2-NEXT:    20090:       addiu   $25, $25, 160
# ORPH2-NEXT:    20094:       nop

# ORPH2: <foo2>:
# ORPH2-NEXT:    200a0:       nop
