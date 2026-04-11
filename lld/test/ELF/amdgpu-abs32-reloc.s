# REQUIRES: amdgpu

# RUN: llvm-mc -filetype=obj -triple=amdgcn-- -mcpu=fiji %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d %t | FileCheck %s
# RUN: llvm-readelf -s -d %t | FileCheck %s --check-prefix=SYMBOL

# CHECK:      <_start>:
# CHECK-NEXT: s_mov_b32 s0, 0xfeedface
# CHECK-NEXT: s_mov_b32 s1, 0xdeadbeef
# CHECK-NEXT: s_endpgm

# SYMBOL: deadbeeffeedface     0 NOTYPE  GLOBAL DEFAULT   ABS sym

.globl sym
sym = 0xdeadbeeffeedface

.globl _start
_start:
  s_mov_b32 s0, sym@abs32@lo
  s_mov_b32 s1, sym@abs32@hi
  s_endpgm
