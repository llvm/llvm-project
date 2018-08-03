# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %s -o %t
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %S/Inputs/hexagon.s -o %t2
# RUN: ld.lld %t2 %t  -o %t3
# RUN: llvm-objdump -d  %t3 | FileCheck %s

# R_HEX_B22_PCREL
call #_start
# CHECK: call 0x11000
