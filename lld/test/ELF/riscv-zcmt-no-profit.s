# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+zcmt,+relax %s -o %t.o
# RUN: ld.lld -e _start --riscv-relax-zcmt %t.o -o %t
# RUN: llvm-readelf -S -s %t | FileCheck /dev/null \
# RUN:   --implicit-check-not=.riscv.jvt --implicit-check-not=__jvt_base$
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases %t | FileCheck %s \
# RUN:   --check-prefix=DIS --implicit-check-not=cm.jt

# DIS-LABEL: <_start>:
# DIS-NEXT: jal zero, {{.*}} <callee>

.attribute arch, "rv64i2p1_zicsr2p0_zca1p0_zcmt1p0"
.option rvc
.globl _start
_start:
  tail callee
  .space 4096
callee:
  ret
