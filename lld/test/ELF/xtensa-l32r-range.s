# REQUIRES: xtensa
## l32r offsets must fall in [-262141, -4]. A forward target is rejected.
## The diagnostic upper bound is uint64_t, so -4 is printed unsigned.
# RUN: llvm-mc -filetype=obj -triple=xtensa -mcpu=esp32 %s -o %t.o
# RUN: not ld.lld %t.o --defsym=far=0x2000 --section-start=.text=0x1000 --image-base=0 -o /dev/null 2>&1 | FileCheck %s

.globl _start
_start:
  l32r a2, far

# CHECK: error: {{.*}}:(.text+0x0): relocation R_XTENSA_SLOT0_OP out of range: 4096 is not in [-262141, -4]; references 'far'
