# REQUIRES: mips
# Check LA25 stubs creation. This stub code is necessary when
# non-PIC code calls PIC function.

# RUN: echo "SECTIONS { \
# RUN:         . = 0x20000; .text ALIGN(0x100) : { *(.text) *(.text.*) } \
# RUN:       }" > %t.script

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r2 \
# RUN:   %p/Inputs/mips-fpic.s -o %t-fpic.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r2 \
# RUN:   %p/Inputs/mips-fnpic.s -o %t-fnpic.o
# RUN: ld.lld -r %t-fpic.o %t-fnpic.o -o %t-sto-pic-r2.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r2 \
# RUN:   %p/Inputs/mips-pic.s -o %t-pic-r2.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r2 \
# RUN:   %s -o %t-npic-r2.o
# RUN: ld.lld %t-npic-r2.o %t-pic-r2.o %t-sto-pic-r2.o \
# RUN:        -script %t.script -o %t-r2.exe
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t-r2.exe \
# RUN:   | FileCheck --check-prefixes=CHECK,R2 %s

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r6 \
# RUN:   %p/Inputs/mips-fpic.s -o %t-fpic.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r6 \
# RUN:   %p/Inputs/mips-fnpic.s -o %t-fnpic.o
# RUN: ld.lld -r %t-fpic.o %t-fnpic.o -o %t-sto-pic-r6.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r6 \
# RUN:   %p/Inputs/mips-pic.s -o %t-pic-r6.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r6 \
# RUN:   %s --defsym MIPSR6=1 --defsym MIPSR6=1 -o %t-npic-r6.o
# RUN: ld.lld %t-npic-r6.o %t-pic-r6.o %t-sto-pic-r6.o \
# RUN:        -script %t.script -o %t-r6.exe
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t-r6.exe \
# RUN:   | FileCheck --check-prefixes=CHECK,R6 %s

# CHECK:     Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT: <__start>:
# R2-NEXT:    20100:       jal     {{.*}} <__LA25Thunk_foo1a>
# R2-NEXT:                 nop
# R2-NEXT:                 jal     {{.*}} <__LA25Thunk_foo2>
# R2-NEXT:                 nop
# R2-NEXT:                 jal     {{.*}} <__LA25Thunk_foo1b>
# R2-NEXT:                 nop
# R2-NEXT:                 jal     {{.*}} <__LA25Thunk_foo2>
# R2-NEXT:                 nop
# R2-NEXT:                 jal     {{.*}} <__LA25Thunk_fpic>
# R2-NEXT:                 nop
# R2-NEXT:                 jal     {{.*}} <fnpic>
# R2-NEXT:                 nop
# R6-NEXT:    20100:       balc     {{.*}} <__LA25Thunk_foo1a>
# R6-NEXT:                 balc     {{.*}} <__LA25Thunk_foo2>
# R6-NEXT:                 balc     {{.*}} <__LA25Thunk_foo1b>
# R6-NEXT:                 balc     {{.*}} <__LA25Thunk_foo2>
# R6-NEXT:                 balc     {{.*}} <__LA25Thunk_fpic>
# R6-NEXT:                 balc     {{.*}} <fnpic>

# CHECK: <__LA25Thunk_fpic>:
# R2:            20130:       lui     $25, 2
# R6:            20120:       aui     $25, $zero, 2
# CHECK-NEXT:                 j       {{.*}} <fpic>
# R2-NEXT:                    addiu   $25, $25, 320
# R6-NEXT:                    addiu   $25, $25, 304
# CHECK-NEXT:                 nop

# CHECK: <fpic>:
# R2-NEXT:    20140:       nop
# R6-NEXT:    20130:       nop

# CHECK: <fnpic>:
# R2-NEXT:    20150:       nop
# R6-NEXT:    20140:       nop

# CHECK: <__LA25Thunk_foo1a>:
# R2:            20154:       lui     $25, 2
# R6:            20144:       aui     $25, $zero, 2
# CHECK:                      j       {{.*}} <foo1a>
# R2-NEXT:                    addiu   $25, $25, 384
# R6-NEXT:                    addiu   $25, $25, 368
# CHECK-NEXT:                 nop

# CHECK: <__LA25Thunk_foo1b>:
# R2:            20164:       lui     $25, 2
# R6:                         aui     $25, $zero, 2
# CHECK-NEXT:                 j       {{.*}} <foo1b>
# R2-NEXT:                    addiu   $25, $25, 388
# R6-NEXT:                    addiu   $25, $25, 372
# CHECK-NEXT:                 nop

# CHECK: <foo1a>:
# R2-NEXT:    20180:       nop
# R6-NEXT:    20170:       nop

# CHECK: <foo1b>:
# R2-NEXT:    20184:       nop
# R6-NEXT:    20174:       nop

# CHECK: <__LA25Thunk_foo2>:
# R2:            20188:       lui     $25, 2
# R6:                         aui     $25, $zero, 2
# CHECK-NEXT:                 j       {{.*}} <foo2>
# R2-NEXT:                    addiu   $25, $25, 416
# R6-NEXT:                    addiu   $25, $25, 400
# CHECK-NEXT:                 nop

# CHECK: <foo2>:
# R2-NEXT:    201a0:       nop
# R6-NEXT:    20190:       nop

  .text
  .globl __start
__start:
.ifdef MIPSR6
  balc foo1a
  balc foo2
  balc foo1b
  balc foo2
  balc fpic
  balc fnpic
.else
  jal foo1a
  jal foo2
  jal foo1b
  jal foo2
  jal fpic
  jal fnpic
.endif
