# REQUIRES: loongarch

# RUN: llvm-mc --filetype=obj --triple=loongarch64 %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s

# CHECK: relocation R_LARCH_ALIGN requires unimplemented linker relaxation

.global _start
_start:
    addi.d $t0, $t0, 1
1:
    nop
    .reloc 1b, R_LARCH_ALIGN, 12
    nop
    nop
