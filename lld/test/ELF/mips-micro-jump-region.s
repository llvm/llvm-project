# REQUIRES: mips
# RUN: llvm-mc -triple=mipsel -mcpu=mips32r2 -mattr=+micromips -filetype=obj %s -o %t.le.o
# RUN: ld.lld -Ttext=0x80100000 %t.le.o -o %t.le
# RUN: llvm-objdump -s -j .text %t.le | FileCheck %s --check-prefix=LE
# RUN: llvm-mc -triple=mips -mcpu=mips32r2 -mattr=+micromips -filetype=obj %s -o %t.be.o
# RUN: ld.lld -Ttext=0x80100000 %t.be.o -o %t.be
# RUN: llvm-objdump -s -j .text %t.be | FileCheck %s --check-prefix=BE

## The jump index occupies 26 bits within the current PC region; the full
## destination address need not fit in a signed 27-bit value.
## Both a tail jump and a call must retain the low 27 address bits.
# LE: 80100000 08d40800 00000000 08f40800 00000000
# BE: 80100000 d4080008 00000000 f4080008 00000000

.set micromips
.set noreorder
.globl __start, target
.type __start,@function
__start:
  j target
  sll $0, $0, 0
  jal target
  sll $0, $0, 0
.type target,@function
target:
  jr $ra
  sll $0, $0, 0
