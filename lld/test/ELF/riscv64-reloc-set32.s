# REQUIRES: riscv
# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t.o
# RUN: ld.lld -Ttext=0x100000000 %t.o -o %t
# RUN: llvm-readelf -x .text %t | FileCheck %s

# CHECK:      section '.text':
# CHECK-NEXT: 0x{{[0-9a-f]+}} 04000000

## R_RISCV_SET32/R_RISCV_SUB32 pairs are used for some CFI label differences.
## Above 4 GiB, the SET32 intermediate value exceeds 32 bits, but the final
## difference is representable.

.globl _start
_start:
  .reloc ., R_RISCV_SET32, .Lend
  .reloc ., R_RISCV_SUB32, _start
  .word 0
.Lend:
