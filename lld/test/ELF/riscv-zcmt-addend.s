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
# CHECK-LABEL: <callee>:

.attribute arch, "rv64i2p1_zicsr2p0_zca1p0_zcmt1p0"
.option rvc
.globl _start
_start:
  .rept 5
  tail callee
  .endr
  .rept 5
1:
  auipc zero, 0
  jalr zero, zero, 0
  .reloc 1b, R_RISCV_CALL_PLT, callee+4
  .reloc 1b, R_RISCV_RELAX
  .endr
  .space 4096
callee:
  nop
  nop
  ret
