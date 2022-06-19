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
# RUN: llvm-objdump -d --no-show-raw-insn %t-r2.exe \
# RUN:   | FileCheck --check-prefixes=CHECK,R2 %s

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r6 \
# RUN:   %p/Inputs/mips-fpic.s -o %t-fpic.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r6 \
# RUN:   %p/Inputs/mips-fnpic.s -o %t-fnpic.o
# RUN: ld.lld -r %t-fpic.o %t-fnpic.o -o %t-sto-pic-r6.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r6 \
# RUN:   %p/Inputs/mips-pic.s -o %t-pic-r6.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r6 \
# RUN:   %s -o %t-npic-r6.o
# RUN: ld.lld %t-npic-r6.o %t-pic-r6.o %t-sto-pic-r6.o \
# RUN:        -script %t.script -o %t-r6.exe
# RUN: llvm-objdump -d --no-show-raw-insn %t-r6.exe \
# RUN:   | FileCheck --check-prefixes=CHECK,R6 %s

# CHECK:     Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT: <__start>:
# CHECK-NEXT:    20100:       jal     {{.*}} <__LA25Thunk_foo1a>
# CHECK-NEXT:                 nop
# CHECK-NEXT:                 jal     {{.*}} <__LA25Thunk_foo2>
# CHECK-NEXT:                 nop
# CHECK-NEXT:                 jal     {{.*}} <__LA25Thunk_foo1b>
# CHECK-NEXT:                 nop
# CHECK-NEXT:                 jal     {{.*}} <__LA25Thunk_foo2>
# CHECK-NEXT:                 nop
# CHECK-NEXT:                 jal     {{.*}} <__LA25Thunk_fpic>
# CHECK-NEXT:                 nop
# CHECK-NEXT:                 jal     {{.*}} <fnpic>
# CHECK-NEXT:                 nop

# CHECK: <__LA25Thunk_fpic>:
# R2:            20130:       lui     $25, 2
# R6:            20130:       aui     $25, $zero, 2
# CHECK-NEXT:                 j       {{.*}} <fpic>
# CHECK-NEXT:                 addiu   $25, $25, 320
# CHECK-NEXT:                 nop

# CHECK: <fpic>:
# CHECK-NEXT:    20140:       nop

# CHECK: <fnpic>:
# CHECK-NEXT:    20150:       nop

# CHECK: <__LA25Thunk_foo1a>:
# R2:            20154:       lui     $25, 2
# R6:            20154:       aui     $25, $zero, 2
# CHECK:                      j       {{.*}} <foo1a>
# CHECK-NEXT:                 addiu   $25, $25, 384
# CHECK-NEXT:                 nop

# CHECK: <__LA25Thunk_foo1b>:
# R2:            20164:       lui     $25, 2
# R6:                         aui     $25, $zero, 2
# CHECK-NEXT:                 j       {{.*}} <foo1b>
# CHECK-NEXT:                 addiu   $25, $25, 388
# CHECK-NEXT:                 nop

# CHECK: <foo1a>:
# CHECK-NEXT:    20180:       nop

# CHECK: <foo1b>:
# CHECK-NEXT:    20184:       nop

# CHECK: <__LA25Thunk_foo2>:
# R2:            20188:       lui     $25, 2
# R6:                         aui     $25, $zero, 2
# CHECK-NEXT:                 j       {{.*}} <foo2>
# CHECK-NEXT:                 addiu   $25, $25, 416
# CHECK-NEXT:                 nop

# CHECK: <foo2>:
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
