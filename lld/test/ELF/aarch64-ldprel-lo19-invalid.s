# REQUIRES: aarch64
# RUN: rm -rf %t && mkdir %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-none %s -o a.o
# RUN: not ld.lld -shared a.o 2>&1 | FileCheck %s --implicit-check-not=error:

# CHECK: error: a.o:(.text+0x0): relocation R_AARCH64_LD_PREL_LO19 out of range: 2131192 is not in [-1048576, 1048575]; references section '.data'

  ldr x8, patatino
  .data
  .zero 2000000
patatino:
