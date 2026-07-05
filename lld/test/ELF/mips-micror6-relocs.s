# REQUIRES: mips

# Check handling of microMIPS R6 relocations.

# RUN: echo "SECTIONS { \
# RUN:         . = 0x20000;  .text ALIGN(0x100) : { *(.text) } \
# RUN:       }" > %t.script

# RUN: llvm-mc -filetype=obj -triple=mips -mcpu=mips32r6 \
# RUN:         %S/Inputs/mips-micro.s -o %t1eb.o
# RUN: llvm-mc -filetype=obj -triple=mips -mcpu=mips32r6 %s -o %t2eb.o
# RUN: ld.lld -o %teb.exe -script %t.script %t1eb.o %t2eb.o
# RUN: llvm-objdump --no-print-imm-hex -d -t --mattr=micromips --no-show-raw-insn %teb.exe \
# RUN:   | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=mipsel -mcpu=mips32r6 \
# RUN:         %S/Inputs/mips-micro.s -o %t1el.o
# RUN: llvm-mc -filetype=obj -triple=mipsel -mcpu=mips32r6 %s -o %t2el.o
# RUN: ld.lld -o %tel.exe -script %t.script %t1el.o %t2el.o
# RUN: llvm-objdump --no-print-imm-hex -d -t --mattr=micromips --no-show-raw-insn %tel.exe \
# RUN:   | FileCheck %s

# CHECK: 00020100 g F     .text  00000000 0x80 foo
# CHECK: 00020110 g       .text  00000000 0x80 __start

# CHECK:      <__start>:
# CHECK-NEXT:    20110:  lapc   $2, -12
# CHECK-NEXT:            beqzc  $3, 0x200f0
# CHECK-NEXT:            balc   {{.*}} <foo>

  .text
  .set micromips
  .global __start
__start:
  addiupc $2, foo+4   # R_MICROMIPS_PC19_S2
  beqzc   $3, foo+4   # R_MICROMIPS_PC21_S1
  balc    foo+4       # R_MICROMIPS_PC26_S1
