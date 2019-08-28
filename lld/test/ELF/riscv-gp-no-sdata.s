# REQUIRES: riscv
# RUN: llvm-mc -filetype=obj -triple=riscv32 %s -o %t.32.o
# RUN: ld.lld -pie %t.32.o -o %t.32
# RUN: llvm-readelf -s %t.32 | FileCheck --check-prefix=SYM %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t.64.o
# RUN: ld.lld -pie %t.64.o -o %t.64
# RUN: llvm-readelf -s %t.64 | FileCheck --check-prefix=SYM %s

## If there is an undefined reference to __global_pointer$ but .sdata doesn't
## exist, define __global_pointer$ and set its st_shndx arbitrarily to 1.

# SYM: {{0*}}00000800 0 NOTYPE GLOBAL DEFAULT 1 __global_pointer$

lla gp, __global_pointer$
