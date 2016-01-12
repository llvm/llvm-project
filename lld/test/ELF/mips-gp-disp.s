# Check that even if _gp_disp symbol is defined in the shared library
# we use our own value.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         %S/Inputs/mips-gp-disp-def.s -o %t-ext.o
# RUN: ld.lld -shared -o %t-ext-int.so %t-ext.o
# RUN: sed -e 's/XXXXXXXX/_gp_disp/g' %t-ext-int.so > %t-ext.so
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld -shared -o %t.so %t.o %t-ext.so
# RUN: llvm-readobj -symbols %t.so | FileCheck -check-prefix=INT-SO %s
# RUN: llvm-readobj -symbols %t-ext.so | FileCheck -check-prefix=EXT-SO %s
# RUN: llvm-objdump -d -t %t.so | FileCheck -check-prefix=DIS %s

# REQUIRES: mips

# INT-SO-NOT:  Name: _gp_disp

# EXT-SO:      Name: _gp_disp
# EXT-SO-NEXT: Value: 0x20010

# DIS:      Disassembly of section .text:
# DIS-NEXT: __start:
# DIS-NEXT:    10000:  3c 08 00 01  lui   $8, 1
# DIS-NEXT:    10004:  21 08 7f f0  addi  $8, $8, 32752
#                                                 ^-- 0x37ff0 & 0xffff
# DIS: 00027ff0  *ABS*  00000000 _gp

  .text
  .globl  __start
__start:
  lui    $t0,%hi(_gp_disp)
  addi   $t0,$t0,%lo(_gp_disp)
  lw     $v0,%call16(_foo)($gp)
