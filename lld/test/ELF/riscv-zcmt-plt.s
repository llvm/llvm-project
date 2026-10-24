# REQUIRES: riscv

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+zcmt,+relax shared.s -o shared.o
# RUN: ld.lld -shared shared.o -o shared.so
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+zcmt,+relax main.s -o main.o
# RUN: ld.lld -e _start --riscv-relax-zcmt main.o shared.so -o main
# RUN: llvm-readelf -S -s main | FileCheck %s --check-prefix=SEC
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases main | FileCheck %s --check-prefix=DIS

# SEC: .riscv.jvt
# SEC: __jvt_base$
# DIS-LABEL: <_start>:
# DIS-NEXT: cm.jt 0
# DIS-NEXT: cm.jt 0
# DIS-NEXT: cm.jt 0
# DIS-NEXT: cm.jt 0
# DIS-NEXT: cm.jt 0

#--- shared.s
.attribute arch, "rv64i2p1_zicsr2p0_zca1p0_zcmt1p0"
.text
.globl foo
foo:
  ret

#--- main.s
.attribute arch, "rv64i2p1_zicsr2p0_zca1p0_zcmt1p0"
.option rvc
.globl _start
_start:
  .rept 5
  tail foo@plt
  .endr
  .space 4096
