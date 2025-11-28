# REQUIRES: riscv

// Check that relaxation correctly adjusts symbol addresses and sizes.

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax -mattr=zcmt %s -o %t.rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax -mattr=zcmt %s -o %t.rv64.o
# RUN: ld.lld -Ttext=0x100000 --relax-tbljal %t.rv32.o -o %t.rv32
# RUN: ld.lld -Ttext=0x100000 --relax-tbljal %t.rv64.o -o %t.rv64

# RUN: llvm-readelf -s %t.rv32 | FileCheck --check-prefix=CHECK32 %s
# RUN: llvm-readelf -s %t.rv64 | FileCheck --check-prefix=CHECK64 %s

# CHECK32:      00100000     4 NOTYPE  LOCAL  DEFAULT     1 a
# CHECK32-NEXT: 00100000     6 NOTYPE  LOCAL  DEFAULT     1 b
# CHECK32-NEXT: 00100000     0 NOTYPE  LOCAL  DEFAULT     1 $x
# CHECK32-NEXT: 00100004     2 NOTYPE  LOCAL  DEFAULT     1 c
# CHECK32-NEXT: 00100004     6 NOTYPE  LOCAL  DEFAULT     1 d
# CHECK32-NEXT: 00100000    10 NOTYPE  GLOBAL DEFAULT     1 _start
# CHECK32-NEXT: 00100040     0 NOTYPE  GLOBAL DEFAULT     2 __jvt_base$

# CHECK64:      00100000     4 NOTYPE  LOCAL  DEFAULT     1 a
# CHECK64-NEXT: 00100000     8 NOTYPE  LOCAL  DEFAULT     1 b
# CHECK64-NEXT: 00100000     0 NOTYPE  LOCAL  DEFAULT     1 $x
# CHECK64-NEXT: 00100004     4 NOTYPE  LOCAL  DEFAULT     1 c
# CHECK64-NEXT: 00100004     8 NOTYPE  LOCAL  DEFAULT     1 d
# CHECK64-NEXT: 00100000    12 NOTYPE  GLOBAL DEFAULT     1 _start
# CHECK64-NEXT: 00100040     0 NOTYPE  GLOBAL DEFAULT     2 __jvt_base$

.global _start
_start:
a:
b:
  add  a0, a1, a2
.size a, . - a
c:
d:
  call _start
.size b, . - b
.size c, . - c
  add a0, a1, a2
.size d, . - d
.size _start, . - _start
