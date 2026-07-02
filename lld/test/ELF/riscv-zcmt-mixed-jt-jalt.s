# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+c,+zcmt,+relax %s -o %t.o
# RUN: ld.lld -e _start --riscv-relax-zcmt %t.o -o %t
# RUN: llvm-readelf -S -s %t | FileCheck %s --check-prefix=SEC
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases %t | FileCheck %s --check-prefix=DIS

# SEC: .riscv.jvt
# SEC-SAME: 000084
# SEC: __jvt_base$
# DIS-LABEL: <_start>:
# DIS-NEXT: cm.jt 0x0
# DIS-NEXT: cm.jt 0x0
# DIS-NEXT: cm.jt 0x0
# DIS-COUNT-67: cm.jalt 0x20
# DIS-NOT: cm.jt
# DIS-NOT: cm.jalt
# DIS-LABEL: <callee>:

.attribute arch, "rv32i2p1_zicsr2p0_zca1p0_zcmt1p0"
.option rvc
.globl _start
_start:
  .rept 3
  tail callee
  .endr
  .rept 67
  call callee
  .endr
  .space 4096
callee:
  ret
