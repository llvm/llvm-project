# REQUIRES: mips
# RUN: rm -rf %t && mkdir %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=mips -mcpu=mips32r6 %s -o a.o
# RUN: llvm-mc -filetype=obj -triple=mips -mcpu=mips32r6 %S/Inputs/mips-align-err.s -o b.o
# RUN: not ld.lld a.o b.o 2>&1 | FileCheck %s --implicit-check-not=error:

# CHECK:      error: a.o:(.text+0x1): unsupported jump/branch instruction between ISA modes referenced by R_MIPS_PC16 relocation
# CHECK-NEXT: error: a.o:(.text+0x1): improper alignment for relocation R_MIPS_PC16: 0xB is not aligned to 4 bytes

        .globl  __start
__start:
.zero 1
        beqc      $5, $6, _foo            # R_MIPS_PC16
