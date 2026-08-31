# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+zcmt,+relax %s -o %t.o
# RUN: ld.lld -pie -e _start --riscv-relax-zcmt %t.o -o %t.pie 2>&1 | FileCheck %s --check-prefix=PIC-WARN
# RUN: ld.lld -shared --riscv-relax-zcmt %t.o -o %t.so 2>&1 | FileCheck %s --check-prefix=PIC-WARN
# RUN: ld.lld -r --riscv-relax-zcmt %t.o -o %t.r 2>&1 | FileCheck %s --check-prefix=REL-WARN
# RUN: llvm-readelf -S -s %t.pie %t.so %t.r | FileCheck /dev/null \
# RUN:   --implicit-check-not=.riscv.jvt --implicit-check-not=__jvt_base$

# PIC-WARN: warning: --riscv-relax-zcmt is disabled for PIE and shared links
# REL-WARN: warning: --riscv-relax-zcmt is disabled for relocatable links

.attribute arch, "rv64i2p1_zicsr2p0_zca1p0_zcmt1p0"
.option rvc
.globl _start
_start:
  .rept 5
  tail callee
  .endr
  .space 4096
callee:
  ret
