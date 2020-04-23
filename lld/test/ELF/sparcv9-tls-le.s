# REQUIRES: sparc
# RUN: llvm-mc -filetype=obj -triple=sparcv9 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=LE %s

## %hix(@tpoff(a)) = ~(st_value(a) - 1026) >> 10 = 1
## %lo(@tpoff(a)) = (st_value(a) - 1026) & 0x3ff | 0x1c00 = -2 (0x1ffe)
# LE:      sethi 1, %o0
# LE-NEXT: xor %o0, -2, %o0
sethi %tle_hix22(a), %o0
xor   %o0, %tle_lox10(a), %o0

.section .tbss
  .globl a
a:
  .zero 1024+2
b:
