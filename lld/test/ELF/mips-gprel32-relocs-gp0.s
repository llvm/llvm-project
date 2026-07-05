# REQUIRES: mips
# Check that relocatable object produced by LLD has zero gp0 value.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld -r -o %t-rel.o %t.o
# RUN: llvm-readobj -A %t-rel.o | FileCheck --check-prefix=REL %s

# RUN: echo "SECTIONS { \
# RUN:         .rodata ALIGN(0x1000) : { *(.rodata) } \
# RUN:         . = 0x20000; .text :  { *(.text) } \
# RUN:       }" > %t.script
# RUN: ld.lld -shared --script %t.script -o %t.so %S/Inputs/mips-gp0-non-zero.o
# RUN: llvm-objdump -s -t %t.so | FileCheck --check-prefix=DUMP %s

# REL: GP: 0x0

# DUMP: SYMBOL TABLE:
# DUMP: 00020008 l       .text          00000000 bar
# DUMP: 00020004 l       .text          00000000 foo
# DUMP: 00028000 l       .got           00000000 .hidden _gp

# DUMP: Contents of section .rodata:
# DUMP: 1000 fffffff4 fffffff8
#            ^ 0x20004 + 0x7ff0 - 0x28000
#                     ^ 0x20008 + 0x7ff0 - 0x28000

  .text
  .global  __start
__start:
  lw      $t0,%call16(__start)($gp)
foo:
  nop
bar:
  nop

  .section .rodata, "a"
v:
  .gpword foo
  .gpword bar
