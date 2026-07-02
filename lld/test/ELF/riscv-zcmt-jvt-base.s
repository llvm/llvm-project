# REQUIRES: riscv

# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+zcmt,+relax duplicate.s -o duplicate.o
# RUN: not ld.lld -e _start --riscv-relax-zcmt duplicate.o -o duplicate 2>&1 | FileCheck %s --check-prefix=DUP

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+zcmt,+relax ref-profit.s -o ref-profit.o
# RUN: ld.lld -e _start --riscv-relax-zcmt ref-profit.o -o ref-profit
# RUN: llvm-readelf -s ref-profit | FileCheck %s --check-prefix=REF --implicit-check-not=__jvt_base$

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+zcmt,+relax ref-noprofit.s -o ref-noprofit.o
# RUN: not ld.lld -e _start --riscv-relax-zcmt ref-noprofit.o -o ref-noprofit 2>&1 | FileCheck %s --check-prefix=REF-NOPROFIT

# DUP: error: duplicate definition of __jvt_base$
# REF: __jvt_base$
# REF-NOPROFIT: error: __jvt_base$ is referenced but no profitable Zcmt jump table was selected

#--- duplicate.s
.attribute arch, "rv64i2p1_zicsr2p0_zca1p0_zcmt1p0"
.option rvc
.globl _start
.globl __jvt_base$
__jvt_base$:
  nop
_start:
  .rept 5
  tail callee
  .endr
  .space 4096
callee:
  ret

#--- ref-profit.s
.attribute arch, "rv64i2p1_zicsr2p0_zca1p0_zcmt1p0"
.option rvc
.globl _start
.globl __jvt_base$
_start:
  la t0, __jvt_base$
  csrw jvt, t0
  .rept 5
  tail callee
  .endr
  .space 4096
callee:
  ret

#--- ref-noprofit.s
.attribute arch, "rv64i2p1_zicsr2p0_zca1p0_zcmt1p0"
.option rvc
.globl _start
.globl __jvt_base$
_start:
  la t0, __jvt_base$
  csrw jvt, t0
  tail callee
  .space 4096
callee:
  ret
