# REQUIRES: loongarch
# RUN: rm -rf %t && split-file %s %t

# RUN: llvm-mc --filetype=obj --triple=loongarch32 %t/32.s -o %t/32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64 %t/64.s -o %t/64.o
# RUN: ld.lld -shared %t/32.o -o %t/32.so
# RUN: llvm-nm %t/32.so | FileCheck --check-prefix=NM32 %s
# RUN: llvm-readobj -r %t/32.so | FileCheck --check-prefix=RELOC32 %s
# RUN: ld.lld -shared %t/64.o -o %t/64.so
# RUN: llvm-nm %t/64.so | FileCheck --check-prefix=NM64 %s
# RUN: llvm-readobj -r %t/64.so | FileCheck --check-prefix=RELOC64 %s

## R_LARCH_32 and R_LARCH_64 are absolute relocation types.
## In PIC mode, they create relative relocations if the symbol is non-preemptable.

# NM32: 000301fc d b
# NM64: 00030350 d b

# RELOC32:      .rela.dyn {
# RELOC32-NEXT:   0x301FC R_LARCH_RELATIVE - 0x301FC
# RELOC32-NEXT:   0x301F8 R_LARCH_32 a 0
# RELOC32-NEXT: }
# RELOC64:      .rela.dyn {
# RELOC64-NEXT:   0x30350 R_LARCH_RELATIVE - 0x30350
# RELOC64-NEXT:   0x30348 R_LARCH_64 a 0
# RELOC64-NEXT: }

#--- 32.s
.globl a, b
.hidden b

.data
.long a
b:
.long b

#--- 64.s
.globl a, b
.hidden b

.data
.quad a
b:
.quad b
