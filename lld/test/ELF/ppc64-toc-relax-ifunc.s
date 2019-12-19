# REQUIRES: ppc

# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t.o
# RUN: echo '.globl ifunc; .type ifunc, %gnu_indirect_function; ifunc:' | \
# RUN:   llvm-mc -filetype=obj -triple=powerpc64le - -o %t1.o
# RUN: ld.lld %t.o %t1.o -o %t
# RUN: llvm-readelf -S -s %t | FileCheck --check-prefix=SEC %s
# RUN: llvm-readelf -x .toc %t | FileCheck --check-prefix=HEX %s
# RUN: llvm-objdump -d %t | FileCheck --check-prefix=DIS %s

## ifunc is a non-preemptable STT_GNU_IFUNC. The R_PPC64_ADDR64 in .toc
## creates a canonical PLT for it and changes its type to STT_FUNC. We can thus
## still perform toc-indirect to toc-relative relaxation because the distance
## to the address of the canonical PLT is fixed.

# SEC: .text PROGBITS 00000000100101e0
# SEC: .plt  NOBITS   00000000100301f8
# SEC: 00000000100101e8 0 FUNC GLOBAL DEFAULT 3 ifunc

## .toc[0] stores the address of the canonical PLT.
# HEX:      section '.toc':
# HEX-NEXT: 0x100201f0 e8010110 00000000

# REL:      .rela.dyn {
# REL-NEXT:   0x100301f8 R_PPC64_IRELATIVE - 0x100101e8
# REL-NEXT: }

# DIS: addi 3, 3,

addis 3, 2, .toc@toc@ha
ld 3, .toc@toc@l(3)

.section .toc,"aw",@progbits
  .quad ifunc
