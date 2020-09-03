# REQUIRES: mips
# Check R_MIPS_GPREL32 relocation calculation.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: echo "SECTIONS { \
# RUN:         .rodata ALIGN(0x1000) : { *(.rodata) } \
# RUN:         . = 0x20000; .text :  { *(.text) } \
# RUN:       }" > %t.script
# RUN: ld.lld -shared --script %t.script -o %t.so %t.o
# RUN: llvm-objdump -s --section=.rodata -t %t.so | FileCheck %s

  .text
  .globl  __start
__start:
  lw      $t0,%call16(__start)($gp)
foo:
  nop
bar:
  nop

  .section .rodata, "a"
v1:
  .gpword foo
  .gpword bar

# CHECK: SYMBOL TABLE:
# CHECK: 00020008 l       .text           00000000 bar
# CHECK: 00020004 l       .text           00000000 foo
# CHECK: 00028000 l       .got            00000000 .hidden _gp

# CHECK: Contents of section .rodata:
# CHECK:  1000 ffff8004 ffff8008
#              ^ 0x20004 - 0x28000
#                       ^ 0x20008 - 0x28000
