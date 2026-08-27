# REQUIRES: sparc
# RUN: llvm-mc -filetype=obj -triple=sparcv9 %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-nm %t.so | FileCheck --check-prefix=NM %s
# RUN: llvm-readobj -r %t.so | FileCheck --check-prefix=RELOC %s
# RUN: ld.lld -shared --apply-dynamic-relocs %t.o -o %t1.so
# RUN: llvm-objdump -s -j .data %t1.so | FileCheck --check-prefix=HEX %s

## R_SPARC_64 is an absolute relocation type.
## In PIC mode, it creates a relative relocation if the symbol is non-preemptable.

# NM: 0000000000300350 d b

# RELOC:      .rela.dyn {
# RELOC-NEXT:   0x300350 R_SPARC_RELATIVE - 0x300350
# RELOC-NEXT:   0x300348 R_SPARC_64 a 0x0
# RELOC-NEXT: }

# HEX:      Contents of section .data:
# HEX-NEXT: 300348 00000000 00000000 00000000 00300350

.globl a, b
.hidden b

.data
.quad a
b:
.quad b
