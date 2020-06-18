# REQUIRES: mips
# Check R_MIPS_HI16 / LO16 relocations calculation against _gp_disp.

# RUN: echo "SECTIONS { \
# RUN:         . = 0x10000; .text ALIGN(0x1000) : { *(.text) } \
# RUN:         . = 0x30000; .got  : { *(.got)  } \
# RUN:       }" > %t.script
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         %S/Inputs/mips-dynamic.s -o %t2.o
# RUN: ld.lld %t1.o %t2.o --script %t.script -o %t.exe
# RUN: llvm-objdump -d -t --no-show-raw-insn %t.exe | FileCheck %s
# RUN: ld.lld %t1.o %t2.o -shared --script %t.script -o %t.so
# RUN: llvm-objdump -d -t --no-show-raw-insn %t.so | FileCheck %s

  .text
  .globl  __start
__start:
  lui    $t0,%hi(_gp_disp)
  addi   $t0,$t0,%lo(_gp_disp)
  lw     $v0,%call16(_foo)($gp)
bar:
  lui    $t0,%hi(_gp_disp)
  addi   $t0,$t0,%lo(_gp_disp)

# CHECK: SYMBOL TABLE:
# CHECK: 0001100c l       .text   00000000 bar
# CHECK: 00037ff0 l       .got    00000000 .hidden _gp
# CHECK: 00011000 g       .text   00000000 __start

# CHECK:      Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT: <__start>:
# CHECK-NEXT:  11000:       lui    $8, 2
#                                      ^-- %hi(0x37ff0-0x11000)
# CHECK-NEXT:  11004:       addi   $8, $8, 28656
#                                          ^-- %lo(0x37ff0-0x11004+4)
# CHECK:      <bar>:
# CHECK-NEXT:  1100c:       lui    $8, 2
#                                      ^-- %hi(0x37ff0-0x1100c)
# CHECK-NEXT:  11010:       addi   $8, $8, 28644
#                                          ^-- %lo(0x37ff0-0x11010+4)
