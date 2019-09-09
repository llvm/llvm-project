# REQUIRES: mips
# Check microMIPS thunk generation.

# RUN: echo "SECTIONS { \
# RUN:         . = 0x20000;  .text ALIGN(0x100) : { *(.text) } \
# RUN:       }" > %t.script

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mcpu=mips32r2 -mattr=micromips %s -o %t-eb.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -position-independent -mcpu=mips32r2 -mattr=micromips \
# RUN:         %S/Inputs/mips-micro.s -o %t-eb-pic.o
# RUN: ld.lld -o %t-eb.exe -script %t.script %t-eb.o %t-eb-pic.o
# RUN: llvm-objdump -d -mattr=+micromips --no-show-raw-insn %t-eb.exe \
# RUN:   | FileCheck --check-prefix=R2 %s

# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux \
# RUN:         -mcpu=mips32r2 -mattr=micromips %s -o %t-el.o
# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux \
# RUN:         -position-independent -mcpu=mips32r2 -mattr=micromips \
# RUN:         %S/Inputs/mips-micro.s -o %t-el-pic.o
# RUN: ld.lld -o %t-el.exe -script %t.script %t-el.o %t-el-pic.o
# RUN: llvm-objdump -d -mattr=+micromips --no-show-raw-insn %t-el.exe \
# RUN:   | FileCheck --check-prefix=R2 %s

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mcpu=mips32r6 -mattr=micromips %s -o %t-eb-r6.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -position-independent -mcpu=mips32r6 -mattr=micromips \
# RUN:         %S/Inputs/mips-micro.s -o %t-eb-pic-r6.o
# RUN: ld.lld -o %t-eb-r6.exe -script %t.script %t-eb-r6.o %t-eb-pic-r6.o
# RUN: llvm-objdump -d -mattr=+micromips --no-show-raw-insn %t-eb-r6.exe \
# RUN:   | FileCheck --check-prefix=R6 %s

# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux \
# RUN:         -mcpu=mips32r6 -mattr=micromips %s -o %t-el-r6.o
# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux \
# RUN:         -position-independent -mcpu=mips32r6 -mattr=micromips \
# RUN:         %S/Inputs/mips-micro.s -o %t-el-pic-r6.o
# RUN: ld.lld -o %t-el-r6.exe -script %t.script %t-el-r6.o %t-el-pic-r6.o
# RUN: llvm-objdump -d -mattr=+micromips --no-show-raw-insn %t-el-r6.exe \
# RUN:   | FileCheck --check-prefix=R6 %s

# R2: __start:
# R2-NEXT:    20100:  jal   131336 <__microLA25Thunk_foo>
# R2-NEXT:            nop

# R2: __microLA25Thunk_foo:
# R2-NEXT:    20108:  lui   $25, 2
# R2-NEXT:            j     131360 <foo>
# R2-NEXT:            addiu $25, $25, 289
# R2-NEXT:            nop

# R6: __start:
# R6-NEXT:    20100:  balc  0 <__start>

# R6: __microLA25Thunk_foo:
# R6-NEXT:    20104:  lui   $25, 2
# R6-NEXT:            addiu $25, $25, 273
# R6-NEXT:            bc    0 <__microLA25Thunk_foo+0x8>

  .text
  .set micromips
  .global __start
__start:
  jal foo
