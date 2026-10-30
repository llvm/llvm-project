# REQUIRES: xtensa
# RUN: llvm-mc -filetype=obj -triple=xtensa -mcpu=esp32 %s -o %t.o
# RUN: ld.lld %t.o --defsym=a=0x2000 --section-start=.CALL=0x1000 --defsym=b=0x40 --defsym=c=0x140 --section-start=.BRANCH=0x5000 --defsym=d=0x5020 --section-start=.BR12=0x100 --image-base=0x0 -o %t
# RUN: llvm-objdump -d --mcpu=esp32 --print-imm-hex %t | FileCheck %s

.section .BR12,"ax",@progbits
 .globl _start
 .balign 0x100
 .type _start,%function
_start:
# CHECK-LABEL:section .BR12
# CHECK:      beqz a2, 0x140
# CHECK-NEXT: bnez a3, 0x140
# CHECK-NEXT: bgez a4, 0x140
# CHECK-NEXT: bltz a5, 0x140
  beqz a2, c
  bnez a3, c
  bgez a4, c
  bltz a5, c

.section .CALL,"ax",@progbits
# CHECK-LABEL: section .CALL:
# CHECK:      call0 0x2000
# CHECK-NEXT: call0 0x2000
# CHECK-NEXT: call0 0x2000
# CHECK-NEXT: call0 0x2000
# CHECK-NEXT: j     0x2000
# CHECK-NEXT: j     0x2000
# CHECK-NEXT: j     0x2000
# CHECK-NEXT: j     0x40
# CHECK-NEXT: j     0x140
# CHECK-NEXT: call0 0x40
# CHECK-NEXT: call0 0x140
# CHECK-NEXT: l32r a3, 0x40
# CHECK-NEXT: callx0 a3
# CHECK-NEXT: l32r a4, 0x140
# CHECK-NEXT: callx0 a4
  call0 a
  call0 a
  call0 a
  call0 a
  j a
  j a
  j a
  j b
  j c
  call0 b
  call0 c
  l32r a3, b
  callx0 a3
  l32r a4, c
  callx0 a4

.section .BRANCH,"ax",@progbits
# CHECK-LABEL: section .BRANCH:
# CHECK:      beq a3, a4, 0x5020
# CHECK-NEXT: ball a3, a4, 0x5020
# CHECK-NEXT: blt a3, a4, 0x5020
# CHECK-NEXT: bt b0, 0x5020
# CHECK-NEXT: beqi a2, 1, 0x5020
# CHECK-NEXT: bgeui a3, 2, 0x5020
# CHECK-NEXT: bf b0, 0x5020
# CHECK-NEXT: beqz.n a2, 0x5020
# CHECK-NEXT: bnez.n a3, 0x5020
  beq a3, a4, d
  ball a3, a4, d
  blt a3, a4, d
  bt b0, d
  beqi a2, 1, d
  bgeui a3, 2, d
  bf b0, d
  beqz.n a2, d
  bnez.n a3, d
