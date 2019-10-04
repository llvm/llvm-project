# REQUIRES: mips
# Check that even if _gp_disp symbol is defined in the shared library
# we use our own value.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: echo "SECTIONS { \
# RUN:         . = 0x1000;  .text ALIGN(0x1000) : { *(.text) } \
# RUN:         . = 0x30000; .got :  { *(.got) } \
# RUN:       }" > %t.script
# RUN: ld.lld -shared --script %t.script -o %t.so %t.o %S/Inputs/mips-gp-disp.so
# RUN: llvm-readelf --symbols %t.so | FileCheck -check-prefix=INT-SO %s
# RUN: llvm-readelf --symbols %S/Inputs/mips-gp-disp.so \
# RUN:   | FileCheck -check-prefix=EXT-SO %s
# RUN: llvm-objdump -d -t --no-show-raw-insn %t.so | FileCheck -check-prefix=DIS %s
# RUN: llvm-readelf -r %t.so | FileCheck -check-prefix=REL %s

# INT-SO: 00000000     0 NOTYPE  LOCAL  HIDDEN   ABS _gp_disp
# EXT-SO: 00020000     0 NOTYPE  GLOBAL DEFAULT    9 _gp_disp

# DIS: 00037ff0  .got   00000000 .hidden _gp
# DIS: 00002000  .text  00000000 __start
# DIS:      Disassembly of section .text:
# DIS-EMPTY:
# DIS-NEXT: __start:
# DIS-NEXT:    lui   $8, 3
# DIS-NEXT:    addi  $8, $8, 24560
#                            ^-- (_gp - __start) & 0xffff

# REL: There are no relocations in this file

  .text
  .globl  __start
__start:
  lui    $t0,%hi(_gp_disp)
  addi   $t0,$t0,%lo(_gp_disp)
  lw     $v0,%call16(_foo)($gp)
