# REQUIRES: mips
# Check LA25 stubs creation. This stub code is necessary when
# non-PIC code calls PIC function.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r2 \
# RUN:   %p/Inputs/mips-fpic.s -o %t-fpic.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r2 \
# RUN:   %p/Inputs/mips-fnpic.s -o %t-fnpic.o
# RUN: ld.lld -r %t-fpic.o %t-fnpic.o -o %t-sto-pic-r2.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r2 \
# RUN:   %p/Inputs/mips-pic.s -o %t-pic-r2.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r2 \
# RUN:   %s -o %t-npic-r2.o
# RUN: ld.lld %t-npic-r2.o %t-pic-r2.o %t-sto-pic-r2.o -o %t-r2.exe
# RUN: llvm-objdump -d --no-show-raw-insn %t-r2.exe | FileCheck --check-prefixes=CHECK,R2 %s

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r6 \
# RUN:   %p/Inputs/mips-fpic.s -o %t-fpic.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r6 \
# RUN:   %p/Inputs/mips-fnpic.s -o %t-fnpic.o
# RUN: ld.lld -r %t-fpic.o %t-fnpic.o -o %t-sto-pic-r6.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r6 \
# RUN:   %p/Inputs/mips-pic.s -o %t-pic-r6.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r6 \
# RUN:   %s -o %t-npic-r6.o
# RUN: ld.lld %t-npic-r6.o %t-pic-r6.o %t-sto-pic-r6.o -o %t-r6.exe
# RUN: llvm-objdump -d --no-show-raw-insn %t-r6.exe | FileCheck --check-prefixes=CHECK,R6 %s

# CHECK:     Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT: __start:
# CHECK-NEXT:    20000:       jal     131120 <__LA25Thunk_foo1a>
# CHECK-NEXT:    20004:       nop
# CHECK-NEXT:    20008:       jal     131160 <__LA25Thunk_foo2>
# CHECK-NEXT:    2000c:       nop
# CHECK-NEXT:    20010:       jal     131136 <__LA25Thunk_foo1b>
# CHECK-NEXT:    20014:       nop
# CHECK-NEXT:    20018:       jal     131160 <__LA25Thunk_foo2>
# CHECK-NEXT:    2001c:       nop
# CHECK-NEXT:    20020:       jal     131188 <__LA25Thunk_fpic>
# CHECK-NEXT:    20024:       nop
# CHECK-NEXT:    20028:       jal     131232 <fnpic>
# CHECK-NEXT:    2002c:       nop
#
# CHECK: __LA25Thunk_foo1a:
# R2:            20030:       lui     $25, 2
# R6:            20030:       aui     $25, $zero, 2
# CHECK:         20034:       j       131152 <foo1a>
# CHECK-NEXT:    20038:       addiu   $25, $25, 80
# CHECK-NEXT:    2003c:       nop

# CHECK: __LA25Thunk_foo1b:
# R2:            20040:       lui     $25, 2
# R6:            20040:       aui     $25, $zero, 2
# CHECK-NEXT:    20044:       j       131156 <foo1b>
# CHECK-NEXT:    20048:       addiu   $25, $25, 84
# CHECK-NEXT:    2004c:       nop

# CHECK: foo1a:
# CHECK-NEXT:    20050:       nop

# CHECK: foo1b:
# CHECK-NEXT:    20054:       nop

# CHECK: __LA25Thunk_foo2:
# R2:            20058:       lui     $25, 2
# R6:            20058:       aui     $25, $zero, 2
# CHECK-NEXT:    2005c:       j       131184 <foo2>
# CHECK-NEXT:    20060:       addiu   $25, $25, 112
# CHECK-NEXT:    20064:       nop

# CHECK: foo2:
# CHECK-NEXT:    20070:       nop

# CHECK: __LA25Thunk_fpic:
# R2:            20074:       lui     $25, 2
# R6:            20074:       aui     $25, $zero, 2
# CHECK-NEXT:    20078:       j       131216 <fpic>
# CHECK-NEXT:    2007c:       addiu   $25, $25, 144
# CHECK-NEXT:    20080:       nop

# CHECK: fpic:
# CHECK-NEXT:    20090:       nop

# CHECK: fnpic:
# CHECK-NEXT:    200a0:       nop

# Make sure the thunks are created properly no matter how
# objects are laid out.
#
# RUN: ld.lld %t-pic-r2.o %t-npic-r2.o %t-sto-pic-r2.o -o %t.exe
# RUN: llvm-objdump -d --no-show-raw-insn %t.exe | FileCheck -check-prefixes=REVERSE,REV-R2 %s
#
# RUN: ld.lld %t-pic-r6.o %t-npic-r6.o %t-sto-pic-r6.o -o %t.exe
# RUN: llvm-objdump -d --no-show-raw-insn %t.exe | FileCheck -check-prefixes=REVERSE,REV-R6 %s

# REVERSE: Disassembly of section .text:
# REVERSE-EMPTY:
# REVERSE-NEXT: __LA25Thunk_foo1a:
# REV-R2:          20000:       lui     $25, 2
# REV-R6:          20000:       aui     $25, $zero, 2
# REVERSE:         20004:       j       131104 <foo1a>
# REVERSE-NEXT:    20008:       addiu   $25, $25, 32
# REVERSE-NEXT:    2000c:       nop

# REVERSE: __LA25Thunk_foo1b:
# REV-R2:          20010:       lui     $25, 2
# REV-R6:          20010:       aui     $25, $zero, 2
# REVERSE:         20014:       j       131108 <foo1b>
# REVERSE-NEXT:    20018:       addiu   $25, $25, 36
# REVERSE-NEXT:    2001c:       nop

# REVERSE: foo1a:
# REVERSE-NEXT:    20020:       nop

# REVERSE: foo1b:
# REVERSE-NEXT:    20024:       nop

# REVERSE: __LA25Thunk_foo2:
# REV-R2:          20028:       lui     $25, 2
# REV-R6:          20028:       aui     $25, $zero, 2
# REVERSE:         2002c:       j       131136 <foo2>
# REVERSE-NEXT:    20030:       addiu   $25, $25, 64
# REVERSE-NEXT:    20034:       nop

# REVERSE: foo2:
# REVERSE-NEXT:    20040:       nop

# REVERSE: __start:
# REVERSE-NEXT:    20050:       jal     131072 <__LA25Thunk_foo1a>
# REVERSE-NEXT:    20054:       nop
# REVERSE-NEXT:    20058:       jal     131112 <__LA25Thunk_foo2>
# REVERSE-NEXT:    2005c:       nop
# REVERSE-NEXT:    20060:       jal     131088 <__LA25Thunk_foo1b>
# REVERSE-NEXT:    20064:       nop
# REVERSE-NEXT:    20068:       jal     131112 <__LA25Thunk_foo2>
# REVERSE-NEXT:    2006c:       nop
# REVERSE-NEXT:    20070:       jal     131200 <__LA25Thunk_fpic>
# REVERSE-NEXT:    20074:       nop
# REVERSE-NEXT:    20078:       jal     131232 <fnpic>
# REVERSE-NEXT:    2007c:       nop

# REVERSE: __LA25Thunk_fpic:
# REV-R2:          20080:       lui     $25, 2
# REV-R6:          20080:       aui     $25, $zero, 2
# REVERSE:         20084:       j       131216 <fpic>
# REVERSE-NEXT:    20088:       addiu   $25, $25, 144
# REVERSE-NEXT:    2008c:       nop

# REVERSE: fpic:
# REVERSE-NEXT:    20090:       nop

# REVERSE: fnpic:
# REVERSE-NEXT:    200a0:       nop

  .text
  .globl __start
__start:
  jal foo1a
  jal foo2
  jal foo1b
  jal foo2
  jal fpic
  jal fnpic
