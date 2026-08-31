# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+zcmt,+relax %s -o %t.o
# RUN: ld.lld -e _start --riscv-relax-zcmt %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases %t | FileCheck %s

# CHECK-LABEL: <_start>:
# CHECK-NEXT: cm.jt 0x0
# CHECK-NEXT: cm.jt 0x0
# CHECK-NEXT: cm.jt 0x0
# CHECK-NEXT: cm.jt 0x0
# CHECK-NEXT: cm.jt 0x0
# CHECK-NEXT: cm.jt 0x1
# CHECK-NEXT: cm.jt 0x1
# CHECK-NEXT: cm.jt 0x1
# CHECK-NEXT: cm.jt 0x1
# CHECK-NEXT: cm.jt 0x1
# CHECK-NOT: cm.jt
# CHECK-LABEL: <callee0>:

.attribute arch, "rv64i2p1_zicsr2p0_zca1p0_zcmt1p0"
.option rvc
.globl _start
_start:
  .rept 5
  tail callee0
  .endr
  .rept 5
  tail callee1
  .endr
  .space 4096
callee0:
  ret
callee1:
  ret
