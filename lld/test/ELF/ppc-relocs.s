# RUN: llvm-mc -filetype=obj -triple=powerpc-unknown-freebsd %s -o %t
# RUN: ld.lld %t -o %t2
# RUN: llvm-objdump -d %t2 | FileCheck %s
# REQUIRES: ppc

.section .R_PPC_ADDR16_HA,"ax",@progbits
.globl _start
_start:
  lis 4, msg@ha
msg:
  .string "foo"
  len = . - msg

# CHECK: Disassembly of section .R_PPC_ADDR16_HA:
# CHECK: _start:
# CHECK:    11000:       3c 80 00 01     lis 4, 1
# CHECK: msg:
# CHECK:    11004:       66 6f 6f 00     oris 15, 19, 28416

.section .R_PPC_ADDR16_LO,"ax",@progbits
  addi 4, 4, msg@l
mystr:
  .asciz "blah"
  len = . - mystr

# CHECK: Disassembly of section .R_PPC_ADDR16_LO:
# CHECK: .R_PPC_ADDR16_LO:
# CHECK:    11008:       38 84 10 04     addi 4, 4, 4100
# CHECK: mystr:
# CHECK:    1100c:       62 6c 61 68     ori 12, 19, 24936
