# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv32 %s -o %t32.o
# RUN: llvm-readobj -r -x .text %t32.o | FileCheck %s --check-prefix=REL
# RUN: ld.lld %t32.o -Ttext=0x80000000 -o %t32
# RUN: llvm-readelf -x .text %t32 | FileCheck %s --check-prefix=HEX

# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t64.o
# RUN: llvm-readobj -r -x .text %t64.o | FileCheck %s --check-prefix=REL
# RUN: ld.lld %t64.o -Ttext=0x100000000 -o %t64
# RUN: llvm-readelf -x .text %t64 | FileCheck %s --check-prefix=HEX

# REL:      .rela.text {
# REL-NEXT:   0x0 R_RISCV_SET32 .Lend 0x0
# REL-NEXT:   0x0 R_RISCV_SUB32 _start 0x0

# HEX:      section '.text':
# HEX-NEXT: 0x{{[0-9a-f]+}} 04000000

.globl _start
_start:
    .reloc ., R_RISCV_SET32, .Lend
    .reloc ., R_RISCV_SUB32, _start
    .word 0
.Lend:
